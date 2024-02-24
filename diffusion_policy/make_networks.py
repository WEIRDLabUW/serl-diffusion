import torch
import torch.nn as nn
from diffusion_policy.networks import ConditionalUnet1D
from diffusers import DDPMScheduler, EMAModel, get_scheduler
from diffusion_policy.dataset import SERLImageDataset
from diffusion_policy.networks import get_resnet, replace_bn_with_gn
from diffusion_policy.configs import DiffusionModelRunConfig
def instantiate_model_artifacts(cfig: DiffusionModelRunConfig, model_only: bool = False):
    '''
    Instantiate the model and the training objects.
    If model only, returns network and scheduler and device only
    If not model only, returns network, ema, noise scheduler, optimizer, lr_scheduler, dataloader, stats, device
    '''
    device = torch.device('cuda')

    vision_feature_dim = 512 * cfg.num_cameras
    vision_encoder = get_resnet('resnet18')
    vision_encoder = replace_bn_with_gn(vision_encoder)

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
    obs_dim = vision_feature_dim + (cfg.state_len if cfg.with_state else 0)

    # create network object
    noise_pred_net = ConditionalUnet1D(
        input_dim=cfg.action_dim,
        global_cond_dim=obs_dim * cfg.obs_horizon
    )

    # the final arch has 2 parts
    nets = nn.ModuleDict({
        'vision_encoder': vision_encoder,
        'noise_pred_net': noise_pred_net
    })
    # device transfer
    _ = nets.to(device)

    if model_only:
        return nets, noise_scheduler, device
    # create dataset from file
    dataset = SERLImageDataset(
        dataset_path=cfg.dataset_path,
        pred_horizon=cfg.pred_horizon,
        obs_horizon=cfg.obs_horizon,
        action_horizon=cfg.action_horizon,
        num_trajectories=cfg.num_trajs,
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
