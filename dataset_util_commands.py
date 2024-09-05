import collections
import json
import pickle
import queue
import time
from itertools import chain
import argparse
import h5py
from dataclasses import dataclass
import tqdm.auto as tqdm
from diffusion_policy.configs import DatasetConfig
from serl_robot_infra.franka_env.envs.franka_env import ImageDisplayer


@dataclass
class DatasetUtilConfig:
    # Merge parameters
    # SERL replay paramaters
    serl_data_path: str = "peg_insert_3_demos_2024-08-23_10-18-32.pkl"

    # Robosuite parameters.
    render: bool = False
    data_config: DatasetConfig = DatasetConfig()
    sim_json_path: str = "./data/square_peg.json"
    dataset_path: str = "./data/image.hdf5"
    output_path: str = "./data/peg_data.pkl"


def robosuite_replay(cfg: DatasetUtilConfig):
    import robosuite as suite
    metadata = json.load(open(cfg.sim_json_path, "r"))
    kwargs = metadata["env_kwargs"]
    if cfg.render:
        kwargs["has_renderer"] = True
    env = suite.make(
        env_name=metadata["env_name"],
        **kwargs
    )
    data = h5py.File(cfg.dataset_path, 'r')['data']
    trajectory_data = []
    for traj_key in tqdm.tqdm(data.keys()):
        traj = data[traj_key]
        extract_data_from_trajectory(traj, env, cfg)
        trajectory_data.append(extract_data_from_trajectory(traj, env, cfg))
    pickle.dump(trajectory_data, open(cfg.output_path, 'wb'))


def extract_data_from_trajectory(traj, env, cfg):
    results = collections.defaultdict(list)
    env.reset()
    env.sim.set_state_from_flattened(traj['states'][0])
    env.sim.forward()
    obs = env._get_observations()
    for i in range(len(traj['actions'])):

        action = traj['actions'][i]
        # Store the data:
        results['action'].append(action)
        # print(obs.keys())
        for key in chain(cfg.data_config.image_keys, cfg.data_config.state_keys):
            results[key].append(obs[key])
        obs, reward, done, _ = env.step(action)
        if cfg.render:
            env.render()
    return dict(results)


def merge_datasets(args):
    dataset = pickle.load(open(args.datasets[0], 'rb'))
    for i in range(1, len(args.datasets)):
        dataset.extend(pickle.load(open(args.datasets[i], 'rb')))
    pickle.dump(dataset, open(args.output, 'wb'))


def replay_vr_data(cfg: DatasetUtilConfig):
    dataset = pickle.load(open(cfg.serl_data_path, 'rb'))

    print(f"Num original trajectories = {sum(1 for i in dataset if i['dones'])}")
    indicies_to_keep = []
    HZ = 10
    # This dataset is a list of dictionaries. Each dictionary has the keys "actions", "observations" and
    # "next_observations". The observation is a dictionary of "state", and then image keys like "wrist_1" and "wrist_2"
    # There is also the "rewards" and "masks" and "dones" keys
    endings = [-1] + [i for i in range(len(dataset)) if dataset[i]["dones"]]

    trajectory_num = 0
    img_queue = queue.Queue()
    displayer = ImageDisplayer(img_queue)
    displayer.start()
    while trajectory_num < len(endings) - 2:
        trajectory = dataset[endings[trajectory_num] + 1:endings[trajectory_num + +1] + 1]
        for i in range(len(trajectory)):
            obs = trajectory[i]["observations"]
            images = obs.copy()
            images.pop("state")
            img_queue.put(images)
            time.sleep(1 / HZ)

        response = input("Press c to continue, press d to delete, press any other key to rewatch:\t\t")

        if response == "c":
            indicies_to_keep.extend(range(endings[trajectory_num] + 1, endings[trajectory_num + 1] + 1))
            trajectory_num += 1
            print("kept trajectory")
        elif response == "d":
            trajectory_num += 1
            print("deleted trajectory")
        else:
            print("rewatching trajectory")
            continue

    dataset = [dataset[i] for i in indicies_to_keep]
    num_traj = sum(1 for i in dataset if i["dones"])

    print(f"saving the file")


    file_name = cfg.serl_data_path.split("/")[-1].split(".")[0]
    file_name = file_name.split("_")
    file_name[0] = f"{num_traj}_filtered"
    new_file_name = f"{'_'.join(file_name)}.pkl"
    pickle.dump(dataset, open(new_file_name, 'wb'))
    print(f"File saved! exiting 🚀")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--robosuite", action="store_true")
    parser.add_argument("--replay", action="store_true")

    # Merge Paramaters
    parser.add_argument("--merge", action="store_true")
    parser.add_argument("--datasets", nargs="+", default=[], help="List of datasets to merge")
    parser.add_argument("--output", default=None, help="Output path for the merged dataset")
    args = parser.parse_args()
    cfg = DatasetUtilConfig()
    if args.robosuite:
        robosuite_replay(cfg)
    elif args.replay:
        replay_vr_data(cfg)
    elif args.replay:
        replay_vr_data(cfg)
    elif args.merge:    
        merge_datasets(args)
    else:
        print("Function not a part of this util function. Exiting...")
