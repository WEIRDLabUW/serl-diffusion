import numpy as np
import torch
import torch.nn as nn
import collections
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from tqdm.auto import tqdm

from diffusion_policy.dataset import normalize_data, unnormalize_data
from diffusion_policy.networks import ConditionalUnet1D, get_resnet, replace_bn_with_gn

import hydra
from dataclasses import dataclass
from diffusion_policy.hydra_utils import ExperimentHydraConfig


### ======================= Hydra Configuration  =============================
# TODO: All of the paramaters of the model are saved in the checkpoint, so we don't need to specify them here

@dataclass
class TrainScriptConfig:
    hydra: ExperimentHydraConfig = ExperimentHydraConfig()
    checkpoint_path: str = "${hydra:runtime.cwd}/${checkpoint_name: ${num_trajs}}"
    with_state: bool = True
    state_len: int = 19
    action_dim: int = 6
    steps_per_traj: int = 200

cs = hydra.core.config_store.ConfigStore.instance()
cs.store(name="train_script_config", node=TrainScriptConfig)


@hydra.main(version_base=None, config_name="train_script_config")
def main(cfg: TrainScriptConfig):
    with_state = cfg.with_state
    state_len = cfg.state_len
    # parameters
    pred_horizon = 16
    obs_horizon = 2
    action_horizon = 8
    num_diffusion_iters = 100
    num_cameras = 2
    action_dim = cfg.action_dim
    max_steps = cfg.steps_per_traj

    vision_encoder = get_resnet('resnet18')
    vision_encoder = replace_bn_with_gn(vision_encoder)

    noise_scheduler = DDPMScheduler(
        num_train_timesteps=num_diffusion_iters,
        # the choise of beta schedule has big impact on performance
        # we found squared cosine works the best
        beta_schedule='squaredcos_cap_v2',
        # clip output to [-1,1] to improve stability
        clip_sample=True,
        # our network predicts noise (instead of denoised action)
        prediction_type='epsilon'
    )

    vision_feature_dim = 512 * num_cameras
    # ResNet18 has output dim of 512
    obs_dim = vision_feature_dim + (state_len if with_state else 0)


    # create network object
    noise_pred_net = ConditionalUnet1D(
        input_dim=action_dim,
        global_cond_dim=obs_dim * obs_horizon
    )

    # the final arch has 2 parts
    nets = nn.ModuleDict({
        'vision_encoder': vision_encoder,
        'noise_pred_net': noise_pred_net
    })

    # device transfer
    device = torch.device('cuda')
    _ = nets.to(device)

    checkpoint = torch.load(cfg.checkpoint_path, map_location='cuda')
    ema_nets = nets
    ema_nets.load_state_dict(checkpoint['state_dict'])
    print('Pretrained weights loaded.')
    stats = checkpoint['stats']

    # env = PushTEnv()
    # use a seed >200 to avoid initial states seen in the training dataset
    # env.seed(100000)
    import gymnasium as gym
    from franka_env.envs.wrappers import (
        GripperCloseEnv,
        Quat2EulerWrapper,
        SERLObsWrapper
    )

    env = gym.make(
        "FrankaPegInsert-Vision-v0",
        fake_env=False,

    )
    env = GripperCloseEnv(env)
    env = Quat2EulerWrapper(env)
    env = SERLObsWrapper(env)

    def process_obs(obs):
        # This function processes the observation such that they can just be fed into the model.
        # It should return a dictionary with the following keys
        # 'embed': The image embeddings
        # 'state': The state of the environment
        # You can change how you get this information depending on the environment.
    
        state = obs['state']
        with torch.no_grad():
            im1 = torch.tensor(obs['wrist_1'], dtype=torch.float32).permute(2, 0, 1).to(device)
            im2 = torch.tensor(obs['wrist_2'], dtype=torch.float32).permute(2, 0, 1).to(device)
            im_stack = torch.stack([im1, im2], dim=0)
            images = ema_nets['vision_encoder'](im_stack).cpu().flatten().numpy()
        return {
            'embed': images,
            'state': state
        }

    def run_one_eval():
        # get first observation
        obs, _ = env.reset()
        obs = process_obs(obs)

        # keep a queue of last 2 steps of observations
        obs_deque = collections.deque(
            [obs] * obs_horizon, maxlen=obs_horizon)
        # save visualization and rewards
        rewards = list()
        done = False
        step_idx = 0

        with tqdm(total=max_steps, desc="Eval Peg Insertion Environment") as pbar:
            while not done:
                B = 1
                # stack the last obs_horizon number of observations
                images = np.stack([x['embed'] for x in obs_deque])
                if with_state:
                    agent_poses = np.stack([x['state'] for x in obs_deque])

                    # normalize observation
                    nagent_poses = normalize_data(agent_poses, stats=stats['state'])
                    nagent_poses = torch.from_numpy(nagent_poses).to(device, dtype=torch.float32)

                # images are already normalized to [0,1]
                nimages = images

                # device transfer
                nimages = torch.from_numpy(nimages).to(device, dtype=torch.float32)
                # (2,3,96,96)
                # (2,2)

                # infer action
                with torch.no_grad():
                    # get image features
                    image_features = nimages  # ema_nets['vision_encoder'](nimages)
                    # (2,1024)

                    # concat with low-dim observations
                    if with_state:
                        obs_features = torch.cat([image_features, nagent_poses], dim=-1)
                    else:
                        obs_features = image_features

                    # reshape observation to (B,obs_horizon*obs_dim)
                    obs_cond = obs_features.unsqueeze(0).flatten(start_dim=1)

                    # initialize action from Guassian noise
                    noisy_action = torch.randn(
                        (B, pred_horizon, action_dim), device=device)
                    naction = noisy_action

                    # init scheduler
                    noise_scheduler.set_timesteps(num_diffusion_iters)

                    for k in noise_scheduler.timesteps:
                        # predict noise
                        noise_pred = ema_nets['noise_pred_net'](
                            sample=naction,
                            timestep=k,
                            global_cond=obs_cond
                        )

                        # inverse diffusion step (remove noise)
                        naction = noise_scheduler.step(
                            model_output=noise_pred,
                            timestep=k,
                            sample=naction
                        ).prev_sample

                # unnormalize action
                naction = naction.detach().to('cpu').numpy()
                # (B, pred_horizon, action_dim)
                naction = naction[0]
                action_pred = unnormalize_data(naction, stats=stats['action'])

                # only take action_horizon number of actions
                start = obs_horizon - 1
                end = start + action_horizon
                action = action_pred[start:end, :]
                # (action_horizon, action_dim)

                # execute action_horizon number of steps
                # without replanning
                for i in range(len(action)):
                    # stepping env
                    obs, reward, done, _, info = env.step(action[i])
                    # save observations
                    obs = process_obs(obs)
                    obs_deque.append(obs)
                    # and reward/vis
                    rewards.append(reward)

                    # update progress bar
                    step_idx += 1
                    pbar.update(1)
                    pbar.set_postfix(reward=reward)
                    if step_idx > max_steps:
                        return
                    if done:
                        if reward > 0:
                            print("Success")
                        return

    for i in range(10):
        run_one_eval()

if __name__ == "__main__":
    main()