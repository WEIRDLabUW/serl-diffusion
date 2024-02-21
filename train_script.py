import os
import time
from typing import List, Union
import numpy as np
import torch
import torch.nn as nn
import collections
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.training_utils import EMAModel
from diffusers.optimization import get_scheduler
from tqdm.auto import tqdm

from diffusion_policy.new_data_loader import SERLImageDataset
from diffusion_policy.networks import ConditionalUnet1D, get_resnet, replace_bn_with_gn

import hydra
from dataclasses import dataclass
from diffusion_policy.hydra_utils import ExperimentHydraConfig


### ======================= Hydra Configuration  =============================


@dataclass
class ImageTrainingScriptConfig:
    hydra: ExperimentHydraConfig = ExperimentHydraConfig()
    num_trajs: int = -1
    batch_size: int = 128
    num_epochs: int = 100
    # These functions are defined in the hydras_utils.py file
    # checkpoint_path: str = "${hydra:runtime.cwd}/${checkpoint_name: ${num_trajs}}"
    checkpoint_path: str = "${hydra:runtime.cwd}/1000_states_checkpoint.pth"
    dataset_path: str = "${hydra:runtime.cwd}/peg_insert_100_demos_2024-02-11_13-35-54.pkl"
    with_state: bool = True
    state_len: int = 19
    action_dim: int = 6
    label_path: Union[str, None] = "${hydra:runtime.cwd}/labels.npy"
    # parameter s
    pred_horizon: int = 16
    obs_horizon: int = 2
    action_horizon: int = 8
    num_diffusion_iters: int = 100
    num_cameras: int = 2

cs = hydra.core.config_store.ConfigStore.instance()
cs.store(name="image_train_script", node=ImageTrainingScriptConfig)


@hydra.main(version_base=None, config_name="image_train_script")
def main(cfg: ImageTrainingScriptConfig):
    num_epochs = cfg.num_epochs
    with_state = cfg.with_state
    state_len = cfg.state_len
    # parameters
    pred_horizon = cfg.pred_horizon
    obs_horizon = cfg.obs_horizon
    action_horizon = cfg.action_horizon
    num_diffusion_iters = cfg.num_diffusion_iters
    num_cameras = cfg.num_cameras
    action_dim = cfg.action_dim


    # create dataset from file
    dataset = SERLImageDataset(
        dataset_path=cfg.dataset_path,
        pred_horizon=pred_horizon,
        obs_horizon=obs_horizon,
        action_horizon=action_horizon,
        num_trajectories=cfg.num_trajs,
        labels=cfg.label_path
    )
    stats = dataset.stats

    # create dataloader
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        num_workers=4,
        shuffle=True,
        # accelerate cpu-gpu transfer
        pin_memory=True,
        # don't kill worker process afte each epoch
        persistent_workers=True
    )

    vision_encoder = get_resnet('resnet18')
    # IMPORTANT!
    # replace all BatchNorm with GroupNorm to work with EMA
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

    # Exponential Moving Average
    # accelerates training and improves stability
    # holds a copy of the model weights
    ema = EMAModel(
        parameters=nets.parameters(),
        power=0.75)

    # Standard ADAM optimizer
    # Note that EMA parameter are not optimized
    optimizer = torch.optim.AdamW(
        params=nets.parameters(),
        lr=1e-4, weight_decay=1e-6)

    # Cosine LR schedule with linear warmup
    lr_scheduler = get_scheduler(
        name='cosine',
        optimizer=optimizer,
        num_warmup_steps=500,
        num_training_steps=len(dataloader) * num_epochs
    )

    with tqdm(range(num_epochs), desc='Epoch',) as tglobal:
        # epoch loop
        for epoch_idx in tglobal:
            epoch_loss = list()
            # batch loop
            with tqdm(dataloader, desc='Batch', leave=False) as tepoch:
                for nbatch in tepoch:
                    # data normalized in dataset
                    # device transfer
                    nimage = nbatch['image'][:, :obs_horizon].to(device)
                    naction = nbatch['action'].to(device)
                    B = nimage.shape[0]


                    image_features = nets['vision_encoder'](
                        nimage.flatten(end_dim=2))
                    # print(nimage.flatten(end_dim=2).shape)
                    # time.sleep(100)
                    image_features = image_features.reshape(
                        B, obs_horizon, vision_feature_dim)
                    # (B,obs_horizon,D)
                    # concatenate vision feature and low-dim obs
                    obs_features = image_features
                    if with_state:
                        nagent_pos = nbatch['state'][:, :obs_horizon].to(device)
                        obs_features = torch.cat([image_features, nagent_pos], dim=-1)

                    obs_cond = obs_features.flatten(start_dim=1)
                    # (B, obs_horizon * obs_dim)

                    # sample noise to add to actions
                    noise = torch.randn(naction.shape, device=device)

                    # sample a diffusion iteration for each data point
                    timesteps = torch.randint(
                        0, noise_scheduler.config.num_train_timesteps,
                        (B,), device=device
                    ).long()

                    # add noise to the clean images according to the noise magnitude at each diffusion iteration
                    # (this is the forward diffusion process)
                    noisy_actions = noise_scheduler.add_noise(
                        naction, noise, timesteps)
                    # predict the noise residual
                    noise_pred = noise_pred_net(
                        noisy_actions, timesteps, global_cond=obs_cond)

                    # L2 loss
                    loss = nn.functional.mse_loss(noise_pred, noise)
                    time.sleep(2)

                    # optimize
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                    # step lr scheduler every batch
                    # this is different from standard pytorch behavior
                    lr_scheduler.step()

                    # update Exponential Moving Average of the model weights
                    ema.step(nets.parameters())

                    # logging
                    loss_cpu = loss.item()
                    epoch_loss.append(loss_cpu)
                    tepoch.set_postfix(loss=loss_cpu)
            tglobal.set_postfix(loss=np.mean(epoch_loss))

    ema_nets = nets
    ema.copy_to(ema_nets.parameters())
    # save checkpoint
    # Create a state dict with the paramaters of the model, and the stats of the dataset
    checkpoint = {
        'state_dict': ema_nets.state_dict(),
        'stats': stats,
        'config': cfg
    }
    torch.save(checkpoint, cfg.checkpoint_path)


if __name__ == "__main__":
    main()
