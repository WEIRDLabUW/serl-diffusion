import cv2
import imageio
import numpy as np


class VideoRecorder:
    '''
    this video recorder like it's obs to come in HWC format!
    '''

    def __init__(self, save_dir, render_size=256, fps=20):
        self.save_dir = save_dir

        self.render_size = render_size
        self.fps = fps
        self.frames = []

    def init(self, obs, enabled=True):
        self.frames = []
        self.enabled = self.save_dir is not None and enabled
        if obs.shape[1] % 16 != 0:
            self.render_size = (obs.shape[1] // 16 + 1) * 16
        self.record(obs)

    def record(self, obs):
        try:
            if self.enabled:
                frame = obs
                if frame.shape[1] % 16 != 0:
                    # resize to multiple of 16
                    frame = cv2.resize(
                        obs,
                        dsize=(self.render_size, self.render_size),
                        interpolation=cv2.INTER_CUBIC
                    )
                # not needed for metaworld frames
                # frame = cv2.resize(obs[-3:].transpose(1, 2, 0),
                #                 dsize=(self.render_size, self.render_size),
                #                 interpolation=cv2.INTER_CUBIC)
                self.frames.append(frame)
        except:
            import pdb;
            pdb.set_trace()

    def save(self, file_name):
        if self.enabled:
            path = file_name
            imageio.mimsave(str(path), self.frames, fps=self.fps)
