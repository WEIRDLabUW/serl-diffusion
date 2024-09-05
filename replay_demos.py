"""
This file allows you to replay actions saved in the .pkl format
"""

import gymnasium as gym
import numpy as np
import pickle as pkl
import time

from franka_env.envs.wrappers import (
    GripperCloseEnv,
    Quat2EulerWrapper,
    SERLObsWrapper
)

if __name__ == "__main__":
    path = 'peg_insert_1_demos_2024-08-12_10-31-50.pkl'

    # Make generic environment
    env = gym.make("FrankaPegInsert-Vision-v0")
    # Add in 7th action for gripper
    env = GripperCloseEnv(env)
    # Observation wrappers
    env = Quat2EulerWrapper(env)
    env = SERLObsWrapper(env)

    # Reset the environment
    env.reset()
    time.sleep(1)

    # Load up all actions and whther or not it is done
    actions = []
    dones = []

    with open(path, 'rb') as f:
        data = pkl.load(f)
        for i in data:
            actions.append(i['actions'])
            dones.append(i['dones'])
    
    if(len(actions) != len(dones)):
        raise Exception("Length of actions and dones do not match")

    # Start replaying the actions and resettign env if done
    for i in range(len(actions)):
        env.step(actions[i])
        if dones[i]:
            print("--------------- finished a episode -------------------")
            print("Resetting")
            env.reset()
            time.sleep(1)

    
    env.reset()