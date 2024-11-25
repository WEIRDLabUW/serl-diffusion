import torch
import torch.nn as nn
from diffusion_policy.networks import ConditionalUnet1D
from diffusers import DDPMScheduler, EMAModel, get_scheduler
from diffusion_policy.dataset import SERLImageDataset, HD5PYDataset, JacobPickleDataset, D4RLDataset, CCILDataset
from diffusion_policy.networks import get_resnet, replace_bn_with_gn
from diffusion_policy.configs import DiffusionModelRunConfig


def instantiate_model_artifacts(cfg: DiffusionModelRunConfig, model_only: bool = False):
    '''
    Instantiate the model and the training objects.
    If model only, returns network and scheduler and device only
    If not model only, returns network, ema, noise scheduler, optimizer, lr_scheduler, dataloader, stats, device
    '''
    device = torch.device('cuda')

    noise_scheduler = DDPMScheduler(
        num_train_timesteps=cfg.num_diffusion_iters,
        # the choise of beta schedule has big impact on performance
        # we found squared cosine works the best
        beta_schedule='squaredcos_cap_v2',
        # clip output to [-1,1] to improve stability
        clip_sample=True,
        # our network predicts noise (instead of denoised action)
        prediction_type='epsilon'
    )

    # ResNet18 has output dim of 512
    obs_dim = 0
    if cfg.with_image:
        obs_dim += 512 * cfg.num_cameras
    if cfg.with_state:
        obs_dim += cfg.state_len

    # create network object
    noise_pred_net = ConditionalUnet1D(
        input_dim=cfg.action_dim,
        global_cond_dim=obs_dim * cfg.obs_horizon
    )

    # the final arch has 2 parts
    nets_dict = {
        'noise_pred_net': noise_pred_net
    }
    if cfg.with_image:
        vision_encoder = get_resnet('resnet18')
        vision_encoder = replace_bn_with_gn(vision_encoder)
        nets_dict["vision_encoder"] = vision_encoder

    nets = nn.ModuleDict(nets_dict)
    # device transfer
    _ = nets.to(device)

    if model_only:
        return nets, noise_scheduler, device
    # create dataset from file
    if cfg.dataset.type == 'SERL':
        dataset = SERLImageDataset(
            dataset_path=cfg.dataset.dataset_path,
            pred_horizon=cfg.pred_horizon,
            obs_horizon=cfg.obs_horizon,
            action_horizon=cfg.action_horizon,
            num_trajectories=cfg.dataset.num_traj,
        )
    elif cfg.dataset.type == 'HDF5':
        dataset = HD5PYDataset(
            dataset_path=cfg.dataset.dataset_path,
            pred_horizon=cfg.pred_horizon,
            obs_horizon=cfg.obs_horizon,
            action_horizon=cfg.action_horizon,
            num_trajectories=cfg.dataset.num_traj,
            state_keys=cfg.dataset.state_keys,
            image_keys=cfg.dataset.image_keys
        )
    elif cfg.dataset.type == 'Jacob':
        dataset = JacobPickleDataset(
            dataset_path=cfg.dataset.dataset_path,
            pred_horizon=cfg.pred_horizon,
            obs_horizon=cfg.obs_horizon,
            action_horizon=cfg.action_horizon,
            num_trajectories=cfg.dataset.num_traj,
            state_keys=cfg.dataset.state_keys,
            image_keys=cfg.dataset.image_keys
        )
    elif cfg.dataset.type == 'D4RL':
        dataset = D4RLDataset(
            dataset_path=cfg.dataset.dataset_path,
            pred_horizon=cfg.pred_horizon,
            obs_horizon=cfg.obs_horizon,
            action_horizon=cfg.action_horizon,
            num_trajectories=cfg.dataset.num_traj,
            image_keys=cfg.dataset.image_keys
        )
    elif cfg.dataset.type == "CCIL":
        dataset = CCILDataset(
            demo_data_path=cfg.dataset.demo_dataset_path,
            aug_data_path=cfg.dataset.aug_dataset_path,
            pred_horizon=cfg.pred_horizon,
            obs_horizon=cfg.obs_horizon,
            action_horizon=cfg.action_horizon,
        )
    else:
        raise ValueError(f"Dataset type {cfg.dataset.type} not recognized. Options are SERL or HDF5 or Jacob or D4RL.)")

    stats = dataset.stats

    # create dataloader
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        num_workers=4,
        shuffle=True,
        # accelerate cpu-gpu transfer
        pin_memory=True,
        # don't kill worker process after each epoch
        persistent_workers=True
    )

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
        num_warmup_steps=(len(dataloader) * cfg.num_epochs) // 10,
        num_training_steps=len(dataloader) * cfg.num_epochs
    )
    return nets, ema, noise_scheduler, optimizer, lr_scheduler, dataloader, stats, device
