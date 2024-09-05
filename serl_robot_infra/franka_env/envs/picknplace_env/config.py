import numpy as np
from franka_env.envs.franka_env import DefaultEnvConfig


class PickNPlaceConfig(DefaultEnvConfig):
    """Set the configuration for FrankaEnv."""
    SERVER_URL: str = "http://127.0.0.1:5000/"
    # REALSENSE_CAMERAS = {
    #     "wrist_1": "123622270810",
    #     "wrist_2": "130322270807",
    #     # "agent_view_1": "213522250963"
    # }
    REALSENSE_CAMERAS = {
        "wrist_1": "130322270807",
        "wrist_2": "123622270810",
        "front_view": "207322251049",
        "side_view": "213522250963"
    }

    TARGET_POSE = None
    RESET_POSE = np.array(
        [
            0.37,
            0.01,
            0.13751238794480508,
            3.157248,
            -0.09709,
            0.1721946
        ]
    )
    REWARD_THRESHOLD = None
    ACTION_SCALE = np.array([.5, 0.5, 1])
    RANDOM_RESET = False
    RANDOM_XY_RANGE = 0.1
    RANDOM_RZ_RANGE = np.pi / 6
    ABS_POSE_LIMIT_LOW = np.array(
        [
            RESET_POSE[0] - 1,
            RESET_POSE[1] - 1,
            RESET_POSE[2] - 0.1,
            RESET_POSE[3] - np.pi / 2,
            RESET_POSE[4] - np.pi / 2,
            RESET_POSE[5] - np.pi / 2,
        ]
    )
    ABS_POSE_LIMIT_HIGH = np.array(
        [
            RESET_POSE[0] + 1,
            RESET_POSE[1] + 1,
            RESET_POSE[2] + 1,
            RESET_POSE[3] + np.pi / 2,
            RESET_POSE[4] + np.pi / 2,
            RESET_POSE[5] + np.pi / 2,
        ]
    )

    COMPLIANCE_PARAM = {
        "translational_stiffness": 2000,
        "translational_damping": 100, # 89
        "rotational_stiffness": 200,  # 150
        "rotational_damping": 10, # 7
        "translational_Ki": 0,
        "translational_clip_x": 0.01, # 0.006
        "translational_clip_y": 0.01, # 0.006
        "translational_clip_z": 0.01, # 0.005
        "translational_clip_neg_x": 0.006,
        "translational_clip_neg_y": 0.006,
        "translational_clip_neg_z": 0.005,
        "rotational_clip_x": 0.05,
        "rotational_clip_y": 0.05,
        "rotational_clip_z": 0.03,
        "rotational_clip_neg_x": 0.05,
        "rotational_clip_neg_y": 0.05,
        "rotational_clip_neg_z": 0.03,
        "rotational_Ki": 0,
    }
    PRECISION_PARAM = {
        "translational_stiffness": 3000,
        "translational_damping": 89,
        "rotational_stiffness": 300,
        "rotational_damping": 9,
        "translational_Ki": 0.1,
        "translational_clip_x": 0.01,
        "translational_clip_y": 0.01,
        "translational_clip_z": 0.01,
        "translational_clip_neg_x": 0.01,
        "translational_clip_neg_y": 0.01,
        "translational_clip_neg_z": 0.01,
        "rotational_clip_x": 0.05,
        "rotational_clip_y": 0.05,
        "rotational_clip_z": 0.05,
        "rotational_clip_neg_x": 0.05,
        "rotational_clip_neg_y": 0.05,
        "rotational_clip_neg_z": 0.05,
        "rotational_Ki": 0.1,
    }
    MAX_EPISODE_STEPS = 1000