import gymnasium as gym
from tqdm import tqdm
import numpy as np
import copy
import pickle as pkl
import datetime

from franka_env.envs.relative_env import RelativeFrame
from franka_env.envs.wrappers import (
    GripperCloseEnv,
    SpacemouseIntervention,
    Quat2EulerWrapper,
    SERLObsWrapper
)



if __name__ == "__main__":
    
    env = gym.make("FrankaPegInsert-Vision-v0")
    env = GripperCloseEnv(env)
    env = SpacemouseIntervention(env)
    # env = RelativeFrame(env)
    env = Quat2EulerWrapper(env)
    env = SERLObsWrapper(env)

    obs, _ = env.reset()

    transitions = []
    success_count = 0
    # Change this to the number of demos you want to record
    success_needed = 100
    pbar = tqdm(total=success_needed)

    while success_count < success_needed:
        next_obs, rew, done, truncated, info = env.step(action=np.zeros((6,)))
        actions = info["intervene_action"]
        # print(actions)
        transition = copy.deepcopy(
            dict(
                observations=obs,
                actions=actions,
                next_observations=next_obs,
                rewards=rew,
                masks=1.0 - done,
                dones=done,
            )
        )
        transitions.append(transition)

        obs = next_obs

        if done:
            obs, _ = env.reset()
            success_count += 1
            pbar.update(1)

    uuid = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    file_name = f"peg_insert_{success_needed}_demos_{uuid}.pkl"
    with open(file_name, "wb") as f:
        pkl.dump(transitions, f)
        print(f"saved {success_needed} demos to {file_name}")
