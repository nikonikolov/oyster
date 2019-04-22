import json
import os
import logging
import shutil
import subprocess
import numpy as np
import gym

from PIL import Image

logger = logging.getLogger(__name__)


class RenderWrapper(gym.Wrapper):

    def __init__(self, env, base_path):
        super().__init__(env)
        self.video_recorder = None
        # base_path = self.env.monitor.config.session_path
        self.video_dir = os.path.join(base_path, "videos")
        self.num_episodes = 0
        self.render_period = 5

        # The other env (train/test) might have already created the directory
        if not os.path.exists(self.video_dir):
            os.makedirs(self.video_dir)

    def reset(self):
        self.num_episodes += 1
        env_output = super().reset()
        self._reset_video_recorder()
        return env_output

    def step(self, action):
        env_output = super().step(action)
        self.video_recorder.capture_frame()
        return env_output

    def _reset_video_recorder(self):
        """Close the current video and open a new one"""

        # Close any existing video recorder
        if self.video_recorder is not None:
            self.video_recorder.close()

        # episode_id = self.env.monitor.num_episodes
        # mode = self.env.monitor.mode
        episode_id = self.num_episodes
        mode = 'train'
        video_file = "{}_video_episode_{:06}".format(mode, episode_id)
        video_file = os.path.join(self.video_dir, video_file)

        # Start recording the next video
        self.video_recorder = VideoRecorder(
            env=self.env,
            path=video_file,
            enabled=self._video_enabled(episode_id),
            metadata={'episode_id': episode_id},
        )
        self.video_recorder.capture_frame()

    def close(self):
        if self.video_recorder is not None:
            self.video_recorder.close()
        return super().close()

    def _video_enabled(self, episode_id):
        if self.render_period <= 0:
            return False
        return episode_id % self.render_period == 0

    def __getattr__(self, name):
        return getattr(self.env, name)


class VideoRecorder:
    """Renders a movie of a rollout, frame by frame."""

    def __init__(self, env, path: str, enabled: bool, metadata: dict = None):
        """
        :param env: Environment to take video of
        :param path: str. Path to the video file
        :param enabled: bool. If False, no video is recorded and no additional computation is done
        :param metadata: dict. Optional contents to save to the metadata file
        """
        self.enabled = enabled

        # Don't bother setting anything else if not enabled
        if not self.enabled:
            return

        if env.metadata.get('semantics.async'):
            logger.info('Disabling video recorder. Cannot render {} in async mode'.format(env))
            return

        if 'rgb_array' not in env.metadata.get('render.modes', []):
            logger.info('Disabling video recorder because {} does not support video mode "rgb_array"'.format(env))
            self.enabled = False
            return

        self.env = env
        self.encoder = None
        self.fps = env.metadata.get('video.frames_per_second', 30)
        self.path = '{}.mp4'.format(path)

        # Metadata
        self.metadata = metadata or {}
        self.metadata_path = '{}.meta.json'.format(path)

        # Status
        self.broken = False
        self.empty = True

    @property
    def functional(self):
        return self.enabled and not self.broken

    def capture_frame(self):
        """Render the given `env` and add the resulting frame to the video."""
        if not self.functional:
            return

        # frame = self.env.render(mode='rgb_array')
        frame = self.env.unwrapped.sim.render(500, 500)[::-1, :, :] #TODO
        
        if frame is None:
            logger.warn('Env returned None on render(). Disabling further rendering for {}'.format(self.path))
            self.broken = True
        else:
            self._encode_image_frame(frame)

    def close(self):
        """Make sure to manually close, or else you will leak the encoder process"""
        if not self.enabled:
            return

        if self.encoder:
            self.encoder.close()
            self.encoder = None
        else:
            self.metadata['empty'] = True

        # If broken, get rid of the output file, otherwise we will leak it.
        if self.broken:
            logger.info('Broken video {}'.format(self.path))

            if os.path.exists(self.path):
                os.remove(self.path)

            self.metadata['broken'] = True

        self.write_metadata()

    def _encode_image_frame(self, frame):
        if not self.encoder:
            self.encoder = ImageEncoder(self.path, frame.shape, self.fps)
            self.metadata['cmdline'] = self.encoder.cmdline

        try:
            self.encoder.capture_frame(frame)
        except (RuntimeError, TypeError) as e:
            logger.warn('Invalid video frame, marking as broken: {}'.format(e))
            self.broken = True
        else:
            self.empty = False

    def write_metadata(self):
        with open(self.metadata_path, 'w') as f:
            json.dump(self.metadata, f)


class ImageEncoder:
    def __init__(self, output_path, frame_shape, fps):
        self.process = None

        # Frame shape should be lines-first, so w and h are swapped
        h, w, c = frame_shape
        if c != 3 and c != 4:
            raise ValueError("Expected frame shape (H,W,3) or (H,W,4), but got {}.".format(frame_shape))
        self.width = w
        self.height = h
        self.alpha_channel = (c == 4)
        self.fps = fps

        if shutil.which('avconv') is not None:
            self.backend = 'avconv'
        elif shutil.which('ffmpeg') is not None:
            self.backend = 'ffmpeg'
        else:
            raise RuntimeError("ImageEncoder needs either ffmpeg or avconv, but neither is available")

        self.start(output_path)

    def start(self, output_path):
        self.cmdline = (
            self.backend,
            '-nostats',
            '-loglevel', 'error',  # suppress warnings
            '-y',
            '-r', '%d' % self.fps,
            # input
            '-f', 'rawvideo',
            '-s:v', '{}x{}'.format(self.width, self.height),
            '-pix_fmt', ('rgb32' if self.alpha_channel else 'rgb24'),
            '-i', '-',  # this used to be /dev/stdin, which is not Windows-friendly
            # output
            '-vf', 'scale=trunc(iw/2)*2:trunc(ih/2)*2',
            '-vcodec', 'libx264',
            '-pix_fmt', 'yuv420p',
            output_path,
        )
        self.process = subprocess.Popen(self.cmdline, stdin=subprocess.PIPE)

    def capture_frame(self, frame):
        if not isinstance(frame, np.ndarray):
            raise TypeError('Expected frame of type np.ndarray, but got {}'.format(type(frame)))
        if frame.shape != self.frame_shape:
            raise RuntimeError("Configured for frame shape {}, but got {}.".format(self.frame_shape, frame.shape))
        if frame.dtype != np.uint8:
            raise TypeError("Expected frame data type np.uint8, but got {}.".format(frame.dtype))

        self.process.stdin.write(frame.tobytes())

    def close(self):
        self.process.stdin.close()
        exit_value = self.process.wait()
        if exit_value != 0:
            logger.error("VideoRecorder encoder exited with status {}".format(exit_value))

    @property
    def frame_shape(self):
        return (self.height, self.width, 3 + self.alpha_channel)
