from random import choice
from time import time
from threading import Lock
from PointBuffer import PointBuffer


class StopSignSolver:
    """
    Class implementing the logic for solving Stop Sign intersection
    """

    def __init__(self, params):
        self.params = params
        self.point_buffer = PointBuffer(buffer_size=self.params["~ss_buffer_length"],
                                        forget_time=self.params["~ss_buffer_forget_time"],
                                        brightness_threshold=self.params["~ss_brightness_threshold"],
                                        max_diff_threshold=self.params["~ss_max_diff_threshold"],
                                        gblur_sigmaX=self.params["~ss_g_blur_sigma_x"],
                                        gblur_sigmaY=self.params["~ss_g_blur_sigma_y"])
        self.buffer_lock = Lock()

        # Initial blinking frequency
        self.blink_freq = 0
        self.begin_blink_time: float = 0  # need to compute for how long the same freq is used
        self.reset()

        # Duration for initial sensing
        self.sensing_duration = self.params["~sensing_duration_sec"]

        # Wait time when conflicting freq is detected with other bots
        self.conflict_wait_with_same_freq_sec = None

        self.img_avrg_fps = self.params["~default_img_fps"] # Start with default FPS

    def update_fps(self, fps):
        self.img_avrg_fps = fps

    def reset(self):
        """
        Resets the solver to it's initial state. Should be called when ever the solving should be restarted
        for example when the intersection type is received/change
        """
        with self.buffer_lock:
            self.point_buffer = PointBuffer(buffer_size=self.params["~ss_buffer_length"],
                                            forget_time=self.params["~ss_buffer_forget_time"],
                                            brightness_threshold=self.params["~ss_brightness_threshold"],
                                            max_diff_threshold=self.params["~ss_max_diff_threshold"],
                                            gblur_sigmaX=self.params["~ss_g_blur_sigma_x"],
                                            gblur_sigmaY=self.params["~ss_g_blur_sigma_y"])

        # Reset blinking frequency
        print(f" reset, OLD self-freq: {self.blink_freq}")
        self.blink_freq = choice(self.params["~permitted_freq_list"])
        print(f" reset, NEW self-freq: {self.blink_freq}")
        self.begin_blink_time = time()

    def get_blink_freq(self):
        """
        Return blinking frequency and color of LED when running this solver
        """
        return self.blink_freq

    def push_camera_image(self, new_img):
        """
        Accumulate the camera images
        """
        with self.buffer_lock:
            self.point_buffer.push_frame(new_img)

    def should_go(self):
        with self.buffer_lock:
            for point in self.point_buffer.points:

                # TODO Tests compare 30 fps to computed img_avrg_fps
                freq_30pfs = point.get_frequency()[0]
                freq = point.get_frequency(self.img_avrg_fps)[0]
                fps = max(self.img_avrg_fps, self.params["~default_img_fps"])
                freq_to_use = point.get_frequency(fps)[0]
                print(f"  ***** OTHERS freq: {freq}, OTHERS freq_30pfs: {freq_30pfs}, OTHERS freq_to_use: {freq_to_use}, SELF freq: {self.blink_freq}, IMG AVG FPS: {self.img_avrg_fps}")

                # TODO Based on some testing with the bots, when the computed FPS (self.img_avrg_fps) is > 30 we get good accurate
                # results to the real frequencies passing the self.img_avrg_fps to point.get_frequency(). However, for some strange reason
                # when the computed FPS is way to low (like 18 FPS) both methods (default 30 vs computed fps) are not accurate but using
                # default 30 FPS get's closer results to the true frequencies.
                # -> Use max (self.img_avrg_fps, 30).
                fps = max(self.img_avrg_fps, self.params["~default_img_fps"])
                freq = point.get_frequency(fps)[0]

                if freq + self.params["~freq_error_upper_margin"] >= self.blink_freq:
                    return False
            return True
