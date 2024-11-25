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
class DatasetConfig:
    # SERL type is the data from record_demos, HDF5 is the type outputed from robomimic, Jacob is the
    # data from process_data.py
    type: str = "CCIL"  # Options are SERL or HDF5 or Jacob or D4RL
    demo_dataset_path: str = "/root/abhayd/CCIL/data/halfcheetah-expert-v2_50.pkl"
    aug_dataset_path: str = "/root/abhayd/CCIL/output/halfcheetah/seed40/halfcheetah-expert-v2/data/backward_euler_fast_none/aug_data.pkl"
    # dataset_path: str = "/root/abhayd/CCIL/output/halfcheetah/seed40/halfcheetah-expert-v2/data/backward_euler_fast_none/aug_data.pkl"
    # dataset_path: str = "${hydra:runtime.cwd}/data/grasp_cube_100_aug.pkl"
    # dataset_path: str = "${hydra:runtime.cwd}/data/lego_filtered.pkl"
    # dataset_path: str = "${hydra:runtime.cwd}/data/coin_filtered.pkl"
    num_traj: int = -1 # Number of trajectories to train on. -1 is all of them

    # The keys from observations that we use for the inputs to our model
    image_keys: list = field(default_factory=lambda: [])

@dataclass
class DiffusionModelRunConfig:
    hydra: ExperimentHydraConfig = ExperimentHydraConfig()
    dataset: DatasetConfig = DatasetConfig()
    device: str = "cuda"
    checkpoint_path: str = "${hydra:runtime.cwd}/halfcheetah_diffusion.pt"

    batch_size: int = 256
    num_epochs: int = 16

    # If with_state, uses the state keys. If without doesn't and state len does not matter
    with_state: bool = True
    # Length of the concatenated state
    state_len: int = 17
    with_image: bool = False
    # Number of images in the observation. Should be equal to the length of image_keys
    num_cameras: int = 0

    action_dim: int = 6
    pred_horizon: int = 16
    obs_horizon: int = 1
    action_horizon: int = 16
    num_diffusion_iters: int = 100
    num_eval_diffusion_iters: int = 16



