import time

import hydra
import numpy as np
import torch
import torch.nn as nn
from tqdm.auto import tqdm

from configs.definitions import DiffusionModelRunConfig
from diffusion_policy.make_networks import instantiate_model_artifacts

import wandb

cs = hydra.core.config_store.ConfigStore.instance()
cs.store(name="diffusion_model_run_cfg", node=DiffusionModelRunConfig)


@hydra.main(version_base=None, config_name="diffusion_model_run_cfg")
def main(cfg: DiffusionModelRunConfig):
    run_training(cfg)


def run_training(cfg: DiffusionModelRunConfig):
    nets, ema, noise_scheduler, optimizer, lr_scheduler, dataloader, stats, device = instantiate_model_artifacts(cfg,
                                                                                                                 model_only=False)
    vision_feature_dim = 512 * cfg.num_cameras
    with tqdm(range(cfg.num_epochs), desc='Epoch', ) as tglobal:
        # epoch loop
        for epoch_idx in tglobal:
            epoch_loss = list()
            # batch loop
            with tqdm(dataloader, desc='Batch', leave=False) as tepoch:
                for nbatch in tepoch:
                    # data normalized in dataset
                    # device transfer
                    nimage = nbatch['image'][:, :cfg.obs_horizon].to(device)
                    naction = nbatch['action'].to(device)
                    B = nimage.shape[0]

                    image_features = nets['vision_encoder'](
                        nimage.flatten(end_dim=2))

                    image_features = image_features.reshape(
                        B, cfg.obs_horizon, vision_feature_dim)
                    # (B,obs_horizon,D)
                    # concatenate vision feature and low-dim obs
                    obs_features = image_features
                    if cfg.with_state:
                        nagent_pos = nbatch['state'][:, :cfg.obs_horizon].to(device)
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
                    noise_pred = nets['noise_pred_net'](
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

def _write_wandb_metrics(self, locs, mean_std, fps):
        iteration = locs['it']
        metrics: Metrics = locs['metrics']
        for value, write_name in zip(metrics.values, metrics.write_names):
            wandb.log({write_name: value}, step=iteration)
        wandb.log({'Loss/learning_rate': self.alg.learning_rate}, step=iteration)


if __name__ == "__main__":
    main()
