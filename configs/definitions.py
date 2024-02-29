from dataclasses import dataclass, field
from typing import Dict

import hydra
from hydra.conf import HydraConf
from omegaconf import OmegaConf

OmegaConf.register_new_resolver("slash_to_dot", lambda dir: dir.replace("/", "."))
OmegaConf.register_new_resolver("checkpoint_name", lambda num_trajs: f"checkpoint_w_{num_trajs}_trajectories.pt")
OmegaConf.register_new_resolver("compute_epochs", lambda num_trajs: (110 - num_trajs) * 10)


@dataclass
class ExperimentHydraConfig(HydraConf):
    root_dir_name: str = "./outputs"
    new_override_dirname: str = "${slash_to_dot: ${hydra:job.override_dirname}}"
    run: Dict = field(default_factory=lambda: {
        # A more sophisticated example:
        # "dir": "${hydra:root_dir_name}/${hydra:new_override_dirname}/seed=${seed}/${now:%Y-%m-%d_%H-%M-%S}",
        # Default behavior logs by date and script name:
        "dir": "${hydra:root_dir_name}/${now:%Y-%m-%d_%H-%M-%S}",
    }
                      )

    sweep: Dict = field(default_factory=lambda: {
        "dir": "${..root_dir_name}/multirun/${now:%Y-%m-%d_%H-%M-%S}",
        "subdir": "${hydra:new_override_dirname}",
    }
                        )

    job: Dict = field(default_factory=lambda: {
        "config": {
            "override_dirname": {
                "exclude_keys": [
                    "sim_device",
                    "rl_device",
                    "headless",
                ]
            }
        },
        "chdir": True
    })

@dataclass
class WandBConfig:
    # settings on logging with Weights and Biases (wandb)
    enable: bool = True # logging with wandb
    project_name: str = 'serl_diffusion' # name of the project to log to
    entity: Optional[str] = None # username for sending the logs, set None for default user in wandb
    log_code: bool = True # code saving (all .py files) to wandb
    codesave_file_extensions: Tuple[str, ...] = ('.py', '.ipynb', '.txt') # extensions of files to save to wandb
    log_model: bool = True # log model checkpoints, same freq. as "RunnerConfig.save_interval"
    log_videos: bool = False # logging of videos on some episodes
    video_frequency: Optional[int] = 10 # how often to save videos, every <frequency> episodes


@dataclass
class DiffusionModelRunConfig:
    hydra: ExperimentHydraConfig = ExperimentHydraConfig()
    device: str = "cuda"
    num_trajs: int = 10
    batch_size: int = 64
    num_epochs: int = 6
    checkpoint_path: str = "${hydra:runtime.cwd}/${checkpoint_name: ${num_trajs}}"
    dataset_path: str = "${hydra:runtime.cwd}/peg_insert_100_demos_2024-02-11_13-35-54.pkl"
    with_state: bool = True
    state_len: int = 19
    action_dim: int = 6
    pred_horizon: int = 16
    obs_horizon: int = 2
    action_horizon: int = 8
    num_diffusion_iters: int = 100
    num_cameras: int = 2





