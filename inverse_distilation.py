import collections
import json
import os
from typing import Callable, Union, Any, Tuple
import robosuite as suite
from dataclasses import dataclass
import numpy as np
import torch
from diffusers import DDPMScheduler
from tqdm import tqdm
from diffusion_policy.policy import DiffusionPolicy
from diffusion_policy.configs import DiffusionModelRunConfig, DatasetConfig
from diffusion_policy.dataset import normalize_data, unnormalize_data, JacobPickleDataset
from diffusion_policy.make_networks import instantiate_model_artifacts
from utils.video_recorder import VideoRecorder
from diffusion_policy.robosuite_models import GaussianActorNetwork

@dataclass
class DistConfig:
    max_steps: int = 400
    expert_model_checkpoint: str = "jacob_dataformat_image_propreo2.pt"
    num_epochs: int = 100


def load_expert_model(checkpoint_path: str) -> Tuple[Any, DDPMScheduler, Any, DiffusionModelRunConfig]:
    checkpoint = torch.load(checkpoint_path, map_location='cuda')
    diff_run_config: DiffusionModelRunConfig = checkpoint['config']
    nets, noise_scheduler, device = instantiate_model_artifacts(diff_run_config, model_only=True)
    nets.load_state_dict(checkpoint['state_dict'])
    print('Pretrained weights loaded.')
    stats = checkpoint['stats']

    return nets, noise_scheduler, stats, diff_run_config


def main(cfg: DistConfig):
    device = "cuda"
    expert_diffusion_model, noise_scheduler, stats, expert_config = load_expert_model(cfg.expert_model_checkpoint)
    dataset = JacobPickleDataset(
        dataset_path="data/peg_data.pkl",
        pred_horizon=expert_config.pred_horizon,
        obs_horizon=expert_config.obs_horizon,
        action_horizon=expert_config.action_horizon,
        num_trajectories=10,
        state_keys=expert_config.dataset.state_keys,
        image_keys=expert_config.dataset.image_keys
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=2,
        num_workers=4,
        shuffle=True,
    )
    # Create the dataset:

    policy = GaussianActorNetwork(
        obs_shapes=collections.OrderedDict({"observation": (512 * expert_config.num_cameras * expert_config.obs_horizon,)}),
        ac_dim=expert_config.action_dim * expert_config.pred_horizon,
        mlp_layer_dims=[1024 * 1, 1024 * 1, 1024 * 1, 1024 * 1]
    ).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-4)


    for batch in dataloader:
        images = batch['image'][:, :expert_config.obs_horizon].to(device)
        expert_actions = batch['action'][:, :expert_config.pred_horizon].to(device)
        B = images.shape[0]
        with torch.no_grad():
            image_features = expert_diffusion_model['vision_encoder'](
                images.flatten(end_dim=2))

        vision_feature_dim = 512 * expert_config.num_cameras

        image_features = image_features.reshape(
            B, expert_config.obs_horizon, vision_feature_dim)
        optimizer.zero_grad()
        policy_dist = policy.jacobian_train_diffusion(expert_diffusion_model, image_features, expert_config, noise_scheduler.betas[0])
        optimizer.step()
        bc_loss = -policy_dist.log_prob(expert_actions.flatten(start_dim=1)).mean()
        print(bc_loss.item())





if __name__ == "__main__":
    main(DistConfig())
