import collections
import json
import time
from typing import Callable
import robosuite as suite
from dataclasses import dataclass
import numpy as np
import torch
from tqdm import tqdm

from diffusion_policy.configs import DiffusionModelRunConfig
from diffusion_policy.dataset import normalize_data, unnormalize_data
from diffusion_policy.make_networks import instantiate_model_artifacts


@dataclass
class EvalConfig:
    num_trajs: int = 100
    render: bool = False
    max_steps: int = 300
    model_checkpoint: str = "outputs/nonweighted_robomimic.pt"
    num_eval_episodes: int = 20



def main(cfg: EvalConfig):
    metadata = json.load(open("data/sim_metadata.json", "r"))
    kwargs = metadata["env_kwargs"]
    if cfg.render:
        kwargs["has_renderer"] = True
    env = suite.make(
        env_name=metadata["env_name"],
        **kwargs
    )
    checkpoint = torch.load(cfg.model_checkpoint, map_location='cuda')
    diff_run_config: DiffusionModelRunConfig = checkpoint['config']

    nets, noise_scheduler, device = instantiate_model_artifacts(diff_run_config, model_only=True)
    nets.load_state_dict(checkpoint['state_dict'])
    print('Pretrained weights loaded.')
    stats = checkpoint['stats']
    successes = 0
    for _ in tqdm(range(cfg.num_eval_episodes), desc='Evaluating'):
        succeeded = run_one_eval(env=env, nets=nets, config=diff_run_config, stats=stats,
                                 noise_scheduler=noise_scheduler,
                                 device=device, max_steps=cfg.max_steps, render=cfg.render)
        if succeeded:
            successes += 1
    # Round to the 3rd decimal place
    success_rate = round(successes / cfg.num_eval_episodes, 3)
    print(f'Success rate: {success_rate}')

def process_obs(obs, nets, device):
    # This function processes the observation such that they can just be fed into the model.
    # It should return a dictionary with the following keys
    # 'embed': The image embeddings
    # 'state': The state of the environment
    # You can change how you get this information depending on the environment.
    state_keys = ['robot0_eef_pos', 'robot0_gripper_qpos', 'robot0_eef_quat']
    state = [obs[state_key] for state_key in state_keys]
    state = np.concatenate(state, axis=-1)
    with torch.no_grad():
        im1 = torch.tensor(obs['agentview_image'], dtype=torch.float32).permute(2, 0, 1).to(device)
        images = nets['vision_encoder'](im1.unsqueeze(0)).cpu().flatten().numpy()
    return {
        'embed': images,
        'state': state
    }


def run_one_eval(env, nets: torch.nn.Module, config: DiffusionModelRunConfig, stats, noise_scheduler, device,
                 max_steps: int, render=True) -> bool:
    # get first observation
    obs = env.reset()
    obs = process_obs(obs, nets, device)

    # keep a queue of last 2 steps of observations
    obs_deque = collections.deque(
        [obs] * config.obs_horizon, maxlen=config.obs_horizon)
    # save visualization and rewards
    rewards = list()
    done = False
    step_idx = 0

    while not done:
        B = 1
        # stack the last obs_horizon number of observations
        images = np.stack([x['embed'] for x in obs_deque])
        if config.with_state:
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
            image_features = nimages
            # (2,1024)

            # concat with low-dim observations
            if config.with_state:
                obs_features = torch.cat([image_features, nagent_poses], dim=-1)
            else:
                obs_features = image_features

            # reshape observation to (B,obs_horizon*obs_dim)
            obs_cond = obs_features.unsqueeze(0).flatten(start_dim=1)

            # initialize action from Gaussian noise
            noisy_action = torch.randn(
                (B, config.pred_horizon, config.action_dim), device=device)
            naction = noisy_action

            # init scheduler
            noise_scheduler.set_timesteps(config.num_diffusion_iters)

            for k in noise_scheduler.timesteps:
                # predict noise
                noise_pred = nets['noise_pred_net'](
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
        start = config.obs_horizon - 1
        end = start + config.action_horizon
        action = action_pred[start:end, :]
        # (action_horizon, action_dim)

        # execute action_horizon number of steps
        # without replanning
        for i in range(len(action)):
            # stepping env
            obs, reward, done, _ = env.step(action[i])
            if render:
                env.render()
            obs = process_obs(obs, nets, device)
            obs_deque.append(obs)
            # and reward/vis
            rewards.append(reward)

            # update progress bar
            step_idx += 1
            if step_idx > max_steps:
                return False
            if done:
                if reward > 0:
                    return True
                return False

if __name__ == "__main__":
    main(EvalConfig())