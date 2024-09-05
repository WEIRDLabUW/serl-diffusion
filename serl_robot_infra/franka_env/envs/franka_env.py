"""Gym Interface for Franka"""
import numpy as np
import gymnasium as gym
import cv2
import copy
from scipy.spatial.transform import Rotation
import time
import requests
import queue
import threading
from datetime import datetime
from collections import OrderedDict

from franka_env.camera.video_capture import VideoCapture
from franka_env.camera.rs_capture import RSCapture
from franka_env.utils.rotations import euler_2_quat, quat_2_euler


class ImageDisplayer(threading.Thread):
    def __init__(self, queue):
        threading.Thread.__init__(self)
        self.queue = queue
        self.daemon = True  # make this a daemon thread

    def run(self):
        while True:
            img_array = self.queue.get()  # retrieve an image from the queue
            if img_array is None:  # None is our signal to exit
                break

            # Extract images
            images = [img_array.get(name) for name in ["wrist_1", "wrist_2", "front_view", "side_view"]]

            # Check for any None values in images
            if any(img is None for img in images):
                continue

            # Resize images to be the same size (assumed to be 480x640 for all images)
            target_size = (640, 480)
            resized_images = [cv2.resize(img, target_size) for img in images]

            # Stack images to create a 2x2 grid
            top_row = np.hstack(resized_images[:2])
            bottom_row = np.hstack(resized_images[2:])
            grid_image = np.vstack([top_row, bottom_row])

            # Display the grid image
            cv2.imshow("RealSense Cameras", grid_image)
            cv2.waitKey(1)


##############################################################################


class DefaultEnvConfig:
    """Default configuration for FrankaEnv. Fill in the values below."""

    SERVER_URL: str = "http://127.0.0.1:5000/"
    REALSENSE_CAMERAS = {
        "wrist_1": "130322274175",
        "wrist_2": "127122270572",
        "front_view": "207322251049",
        "side_view": "213522250963"
    }
    TARGET_POSE: np.ndarray = np.zeros((6,))
    REWARD_THRESHOLD: np.ndarray = np.zeros((6,))
    ACTION_SCALE = np.zeros((3,))
    RESET_POSE = np.zeros((6,))
    RANDOM_RESET = (False,)
    RANDOM_XY_RANGE = (0.0,)
    RANDOM_RZ_RANGE = (0.0,)
    ABS_POSE_LIMIT_HIGH = np.zeros((6,))
    ABS_POSE_LIMIT_LOW = np.zeros((6,))
    COMPLIANCE_PARAM = {}
    PRECISION_PARAM = {}
    MAX_EPISODE_STEPS = 100


##############################################################################


class FrankaEnv(gym.Env):
    def __init__(
        self,
        hz=10,
        fake_env=False,
        save_video=False,
        config: DefaultEnvConfig = None,
    ):
        self.action_scale = config.ACTION_SCALE
        self._TARGET_POSE = config.TARGET_POSE
        self._REWARD_THRESHOLD = config.REWARD_THRESHOLD
        self._max_steps = config.MAX_EPISODE_STEPS
        self.url = config.SERVER_URL
        self.config = config

        # convert last 3 elements from euler to quat, from size (6,) to (7,)
        self.resetpos = np.concatenate(
            [config.RESET_POSE[:3], euler_2_quat(config.RESET_POSE[3:])]
        )

        self.currpos = self.resetpos.copy()
        self.currvel = np.zeros((6,))
        self.q = np.zeros((7,))
        self.dq = np.zeros((7,))
        self.currforce = np.zeros((3,))
        self.currtorque = np.zeros((3,))
        self.currjacobian = np.zeros((6, 7))
        self.closed_gripper = False

        self.curr_gripper_pos = 0
        self.lastsent = time.time()
        self.randomreset = config.RANDOM_RESET
        self.random_xy_range = config.RANDOM_XY_RANGE
        self.random_rz_range = config.RANDOM_RZ_RANGE
        self.hz = hz
        self.joint_reset_cycle = 200  # reset the robot joint every 200 cycles

        if save_video:
            print("Saving videos!")
        self.save_video = save_video
        self.recording_frames = []

        # boundary box
        self.xyz_bounding_box = gym.spaces.Box(
            config.ABS_POSE_LIMIT_LOW[:3],
            config.ABS_POSE_LIMIT_HIGH[:3],
            dtype=np.float64,
        )
        self.rpy_bounding_box = gym.spaces.Box(
            config.ABS_POSE_LIMIT_LOW[3:],
            config.ABS_POSE_LIMIT_HIGH[3:],
            dtype=np.float64,
        )
        # Action/Observation Space
        self.action_space = gym.spaces.Box(
            np.ones((7,), dtype=np.float32) * -1,
            np.ones((7,), dtype=np.float32),
        )

        self.observation_space = gym.spaces.Dict(
            {
                "state": gym.spaces.Dict(
                    {
                        "tcp_pose": gym.spaces.Box(
                            low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32
                        ),  # xyz + quat
                        "tcp_vel": gym.spaces.Box(
                            low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32
                        ),
                        "gripper_pose": gym.spaces.Box(
                            low=-1, high=1, shape=(1,), dtype=np.float32
                        ),
                        "tcp_force": gym.spaces.Box(
                            low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32
                        ),
                        "tcp_torque": gym.spaces.Box(
                            low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32
                        ),
                    }
                ),
                "images": gym.spaces.Dict(
                    {
                        "wrist_1": gym.spaces.Box(
                            low=0, high=255, shape=(480, 640, 3), dtype=np.uint8
                        ),
                        "wrist_2": gym.spaces.Box(
                            low=0, high=255, shape=(480, 640, 3), dtype=np.uint8
                        ),
                        "front_view": gym.spaces.Box(
                            low=0, high=255, shape=(480, 640, 3), dtype=np.uint8
                        ),
                        "side_view": gym.spaces.Box(
                            low=0, high=255, shape=(480, 640, 3), dtype=np.uint8
                        ),
                    }
                ),
            }
        )
        self.cycle_count = 0

        if fake_env:
            return

        self.cap = None
        self.init_cameras(config.REALSENSE_CAMERAS)
        self.img_queue = queue.Queue()
        self.displayer = ImageDisplayer(self.img_queue)
        self.displayer.start()
        print("Initialized Franka")

    def recover(self):
        requests.post(self.url + "clearerr")

    def _send_pos_command(self, pos):
        self.recover()
        arr = np.array(pos).astype(np.float32)
        data = {"arr": arr.tolist()}
        requests.post(self.url + "pose", json=data)

    def send_gripper_command(self, pos, mode="binary"):
        if mode == "binary":
            if (pos >= -1) and (pos <= -0.9) and not self.closed_gripper:  # close gripper
                requests.post(self.url + "close_gripper")
                self.closed_gripper = True
            elif (pos >= 0.9) and (pos <= 1) and self.closed_gripper:  # open gripper
                requests.post(self.url + "open_gripper")
                self.closed_gripper = False
            else:  # do nothing to the gripper
                return
        elif mode == "continuous":
            raise NotImplementedError("Continuous gripper control is optional")

    def update_currpos(self):
        ps = requests.post(self.url + "getstate").json()
        self.currpos[:] = np.array(ps["pose"])
        self.currvel[:] = np.array(ps["vel"])

        self.currforce[:] = np.array(ps["force"])
        self.currtorque[:] = np.array(ps["torque"])
        self.currjacobian[:] = np.reshape(np.array(ps["jacobian"]), (6, 7))

        self.q[:] = np.array(ps["q"])
        self.dq[:] = np.array(ps["dq"])

        self.curr_gripper_pos = np.array(ps["gripper_pos"])

    def clip_safety_box(self, pose):
        """Clip the pose to be within the safety box."""
        pose[:3] = np.clip(
            pose[:3], self.xyz_bounding_box.low, self.xyz_bounding_box.high
        )
        euler = Rotation.from_quat(pose[3:]).as_euler("xyz")

        # Clip first euler angle separately due to discontinuity from pi to -pi
        sign = np.sign(euler[0])
        euler[0] = sign * (
            np.clip(
                np.abs(euler[0]),
                self.rpy_bounding_box.low[0],
                self.rpy_bounding_box.high[0],
            )
        )

        euler[1:] = np.clip(
            euler[1:], self.rpy_bounding_box.low[1:], self.rpy_bounding_box.high[1:]
        )
        pose[3:] = Rotation.from_euler("xyz", euler).as_quat()

        return pose

    def step(self, action):
        start_time = time.time()
        action = np.clip(action, self.action_space.low, self.action_space.high)
        xyz_delta = action[:3]

        self.nextpos = self.currpos.copy()
        self.nextpos[:3] = self.nextpos[:3] + xyz_delta * self.action_scale[0]

        # GET ORIENTATION FROM ACTION
        self.nextpos[3:] = (
            Rotation.from_euler("xyz", action[3:6] * self.action_scale[1])
            * Rotation.from_quat(self.currpos[3:])
        ).as_quat()

        gripper_action = action[6] * self.action_scale[2]

        self.send_gripper_command(gripper_action)
        self._send_pos_command(self.clip_safety_box(self.nextpos))

        self.curr_path_length += 1
        dt = time.time() - start_time
        time.sleep(max(0, (1.0 / self.hz) - dt))

        self.update_currpos()
        ob = self._get_obs()
        reward = self.compute_reward(ob)
        if reward:
            print("Goal reached!")
        done = self.curr_path_length >= self._max_steps or reward
        return ob, int(reward), done, False, {}

    def compute_reward(self, obs):
        current_pose = obs["state"]["tcp_pose"]
        # convert from quat to euler first
        euler_angles = quat_2_euler(current_pose[3:])
        euler_angles = np.abs(euler_angles)
        current_pose = np.hstack([current_pose[:3], euler_angles])
        delta = np.abs(current_pose - self._TARGET_POSE)
        if np.all(delta < self._REWARD_THRESHOLD):
            return True
        else:
            # print(f'Goal not reached, the difference is {delta}, the desired threshold is {_REWARD_THRESHOLD}')
            return False

    def crop_image(self, name, image):
        """Crop realsense images to be a square."""
        if name == "wrist_1":
            return image[:, 80:560, :]
        elif name == "wrist_2":
            return image[:, 80:560, :]
        elif name == "front_view":
            return image[80:560, :, :]
        elif name == "side_view":
            return image[:, 80:560, :]
        # elif name == "agent_view_1":
        #     # The image is a 1280 × 800 image, crop to 800x800
        #     return image[:, 240:1040, :]
        else:
            return ValueError(f"Camera {name} not recognized in cropping")

    
    def get_im(self):
        images = {}
        display_images = {}

        # Define the target size for all images
        target_width = 640
        target_height = 480

        for key, cap in self.cap.items():
            try:
                rgb = cap.read()
                cropped_rgb = self.crop_image(key, rgb)
                resized = cv2.resize(cropped_rgb, (target_width, target_height))  # Resize to target size
                images[key] = resized[..., ::-1]  # Convert BGR to RGB
                display_images[key] = resized
                display_images[key + "_full"] = cropped_rgb
            except Exception as e:
                print(f"Error with {key} camera: {e}")
                cap.close()
                self.init_cameras(self.config.REALSENSE_CAMERAS)
                return self.get_im()

        # Debugging: Print the sizes of the images
        for key, img in display_images.items():
            print(f"{key} size: {img.shape}")

        # Ensure all images are resized to the same dimensions before concatenation
        try:
            resized_full_images = []
            for key in self.cap:
                img = display_images[f"{key}_full"]
                if img.shape[:2] != (target_height, target_width):
                    # Resize the image to fit the target size while preserving aspect ratio
                    aspect_ratio = img.shape[1] / img.shape[0]
                    if aspect_ratio > target_width / target_height:
                        # Wider than target aspect ratio
                        new_width = target_width
                        new_height = int(target_width / aspect_ratio)
                    else:
                        # Taller than target aspect ratio
                        new_height = target_height
                        new_width = int(target_height * aspect_ratio)

                    resized = cv2.resize(img, (new_width, new_height))
                    
                    # Add padding to ensure the image fits the target size
                    top_padding = (target_height - new_height) // 2
                    bottom_padding = target_height - new_height - top_padding
                    left_padding = (target_width - new_width) // 2
                    right_padding = target_width - new_width - left_padding

                    padded_img = cv2.copyMakeBorder(
                        resized,
                        top_padding,
                        bottom_padding,
                        left_padding,
                        right_padding,
                        borderType=cv2.BORDER_CONSTANT,
                        value=[0, 0, 0]  # Padding color (black)
                    )
                else:
                    padded_img = img

                resized_full_images.append(padded_img)

            # Concatenate images along the vertical axis (axis=0)
            concatenated_image = np.concatenate(resized_full_images, axis=0)
            self.recording_frames.append(concatenated_image)
        except ValueError as e:
            print(f"Error concatenating images: {e}")
            for key, img in display_images.items():
                print(f"Image {key} shape: {img.shape}")

        self.img_queue.put(display_images)
        return images





    def _get_state(self):
        state_observation = {
            "tcp_pose": self.currpos,
            "tcp_vel": self.currvel,
            "gripper_pose": self.curr_gripper_pos,
            "tcp_force": self.currforce,
            "tcp_torque": self.currtorque,
        }
        return state_observation

    def _get_obs(self):
        images = self.get_im()
        state_observation = self._get_state()

        observation = {
            "images": images,
            "state": state_observation
        }
        # return copy.deepcopy(dict(images=images, state=state_observation))
        return observation


    def interpolate_move(self, goal, timeout):
        steps = int(timeout * self.hz)
        self.update_currpos()
        path = np.linspace(self.currpos, goal, steps)
        for p in path:
            self._send_pos_command(p)
            time.sleep(1 / self.hz)
        self.update_currpos()

    def go_to_rest(self, joint_reset=False):
        """
        The concrete steps to perform reset should be
        implemented each subclass for the specific task.
        """
        raise NotImplementedError

    def reset(self, joint_reset=False, **kwargs):
        requests.post(self.url + "update_param", json=self.config.COMPLIANCE_PARAM)
        if self.save_video:
            self.save_video_recording()

        self.cycle_count += 1
        if self.cycle_count % self.joint_reset_cycle == 0:
            self.cycle_count = 0
            joint_reset = True

        self.go_to_rest(joint_reset=joint_reset)
        self.send_gripper_command(1)
        self.recover()
        self.curr_path_length = 0

        self.update_currpos()
        o = self._get_obs()

        return o, {}

    def save_video_recording(self):
        try:
            if len(self.recording_frames):
                video_writer = cv2.VideoWriter(
                    f'./videos/{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.mp4',
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    10,
                    self.recording_frames[0].shape[:2][::-1],
                )
                for frame in self.recording_frames:
                    video_writer.write(frame)
                video_writer.release()
            self.recording_frames.clear()
        except Exception as e:
            print(f"Failed to save video: {e}")

    def init_cameras(self, name_serial_dict=None):
        """Init all cameras."""
        if self.cap is not None:  # close cameras if they are already open
            self.close_cameras()
        self.cap = OrderedDict()
        for cam_name, cam_serial in name_serial_dict.items():
            print(f"Next camera = {cam_name}, {cam_serial}")
            cap = VideoCapture(
                RSCapture(name=cam_name, serial_number=cam_serial, depth=False)
            )
            self.cap[cam_name] = cap



    def close_cameras(self):
        """Close both wrist cameras."""
        try:
            for cap in self.cap.values():
                cap.close()
        except Exception as e:
            print(f"Failed to close cameras: {e}")