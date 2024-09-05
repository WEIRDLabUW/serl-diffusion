"""
This file is a demo collection script that you can use on the realworld franka
in order to collect demos for imitation learning.
"""

import gymnasium as gym
from utils.oculus import VRController
from tqdm import tqdm
import copy
import pickle as pkl
import datetime
from gymnasium.spaces import flatten
import argparse


from franka_env.envs.relative_env import RelativeFrame
from franka_env.envs.wrappers import (
    GripperCloseEnv,
    SpacemouseIntervention,
    Quat2EulerWrapper,
    SERLObsWrapper
)


def _flatten_obs(obs, env):
    obs = {
        "state": flatten(env.observation_space["state"], obs["state"]),
        **(obs["images"]),
    }
    return obs


def main(arg):
    env = gym.make("FrankaPickNPlace-Vision-v0")
    # env = gym.make("FrankaPegInsert-Vision-v0")
    # env = gym.make("FrankaPickNPlace-Vision-v0")
    if not arg.gripper:
        env = GripperCloseEnv(env)
    # env = RelativeFrame(env)
    env = Quat2EulerWrapper(env)
    # env = SERLObsWrapper(env)
    obs, _ = env.reset()


    transitions = []
    success_count = 0
    # Change this to the number of demos you want to record
    success_needed = 3
    pbar = tqdm(total=success_needed)
    controller = VRController()
    current_trajectory = []

    while success_count < success_needed:
        action = controller.forward(obs)
        if not arg.gripper:
            action = action[:6]

        next_obs, rew, done, truncated, info = env.step(action=action)
        actions = action
        transition = copy.deepcopy(
            dict(
                observations=_flatten_obs(obs, env) if arg.flatten else obs,
                actions=actions,
                next_observations=_flatten_obs(next_obs, env) if arg.flatten else next_obs,
                rewards=rew,
                masks=1.0 - done,
                dones=done,
            )
        )
        current_trajectory.append(transition)

        obs = next_obs
        done = done or controller.check_done()

        if done:
            _ = env.reset()
            if controller.save_demo():
                success_count += 1
                pbar.update(1)

                # set the reward to 1 and the dones to 1
                current_trajectory[-1]["dones"] = True
                current_trajectory[-1]["rewards"] = 1

                transitions.extend(current_trajectory)
                print(f"Saving trajectory of length {len(current_trajectory)}")
            current_trajectory = []
            obs, _ = env.reset()


    uuid = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    file_name = f"peg_insert_{success_needed}_demos_{uuid}.pkl" if arg.name is None else arg.name
    with open(file_name, "wb") as f:
        pkl.dump(transitions, f)
        print(f"saved {success_needed} demos to {file_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-g", "--gripper", default=True, help="True means with gripper, false means without gripper")
    parser.add_argument("-n", "--name", default=None, help="This is the name that you want to save the demos as. If "
                                                           "you leave it blank, it will save it with the number of "
                                                           "demos and the date and time")
    parser.add_argument("-f", "--flatten", default=True, help="Whether or not to flatten the state obs into one key "
                                                              "of state")
    args = parser.parse_args()
    print(args)
    main(args)
