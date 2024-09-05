import numpy as np
import pyrealsense2 as rs
import time
import cv2

class RSCapture:
    def __init__(self, name, serial_number, dim=(640, 480), depth=False):
        self.name = name
        self.serial_number = serial_number
        self.depth = depth

        # Create a pipeline and configure it
        self.pipe = rs.pipeline()
        self.cfg = rs.config()

        # Configure the camera settings based on the serial number
        if serial_number == "213522250963":  # Side camera
            self.cfg.enable_device(self.serial_number)
            self.cfg.enable_stream(rs.stream.depth, 424, 240, rs.format.z16, 5)
            self.cfg.enable_stream(rs.stream.color, 480, 270, rs.format.rgb8, 5)
        elif serial_number == "207322251049":  # Front camera
            self.cfg.enable_device(self.serial_number)
            self.cfg.enable_stream(rs.stream.depth, 640, 360, rs.format.z16, 15)
            self.cfg.enable_stream(rs.stream.color, 640, 360, rs.format.rgb8, 15)
        elif serial_number == "130322270807":  # D405 camera 1
            self.cfg.enable_device(self.serial_number)
            self.cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 15)
            self.cfg.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 15)
        elif serial_number == "123622270810":  # D405 camera 2
            self.cfg.enable_device(self.serial_number)
            self.cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 15)
            self.cfg.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 15)
        else:
            raise ValueError(f"Unsupported serial number: {serial_number}")
        
        # cv2.cvtColor(color_image_1, cv2.COLOR_BGR2RGB)

        # Start the pipeline
        self.profile = self.pipe.start(self.cfg)

        # Create an align object
        align_to = rs.stream.color
        self.align = rs.align(align_to)

        # Configure depth sensor options if depth is enabled
        if self.depth:
            depth_sensor = self.profile.get_device().first_depth_sensor()
            if depth_sensor:
                visual_preset = 1
                max_measurable_range = 10000.0
                desired_max_range = 5000.0
                depth_units = (max_measurable_range / desired_max_range) * 0.001

                retry_count = 10
                for _ in range(retry_count):
                    try:
                        depth_sensor.set_option(rs.option.visual_preset, visual_preset)
                        depth_sensor.set_option(rs.option.depth_units, depth_units)
                        break
                    except RuntimeError as e:
                        print(f"Error setting options for {self.serial_number}: {e}")
                        time.sleep(2)
                else:
                    print(f"Failed to set options for {self.serial_number}")

    def read(self):
        frames = self.pipe.wait_for_frames(timeout_ms=10000)
        aligned_frames = self.align.process(frames)
        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame() if self.depth else None

        if color_frame and color_frame.is_video_frame():
            image = np.asarray(color_frame.get_data())
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Use cv2 to convert color space
            if self.depth and depth_frame and depth_frame.is_depth_frame():
                depth = np.expand_dims(np.asarray(depth_frame.get_data()), axis=2)
                return True, np.concatenate((image, depth), axis=-1)
            else:
                return True, image
        else:
            return False, None

    def close(self):
        self.pipe.stop()
        self.cfg.disable_all_streams()
