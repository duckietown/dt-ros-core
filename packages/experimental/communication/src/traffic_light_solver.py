import numpy as np
from PointBuffer import PointBuffer
from UtilStates import TrafficLightState
from threading import Lock
from time import time


class TrafficLightSolver:
    """
    Class implementing the logic for solving Traffic Light intersection
    """

    def __init__(self, params):
        self.params = params

        self.buffer = PointBuffer(buffer_size=self.params["~tl_buffer_length"],
                                  forget_time=self.params["~tl_buffer_forget_time"],
                                  brightness_threshold=self.params["~tl_brightness_threshold"],
                                  max_diff_threshold=self.params["~tl_max_diff_threshold"],
                                  gblur_sigmaX=self.params["~tl_g_blur_sigma_x"],
                                  gblur_sigmaY=self.params["~tl_g_blur_sigma_y"])
        self.buffer_lock = Lock()
        self.tl_state = TrafficLightState.Sensing
        self.img_avrg_fps = self.params["~default_img_fps"] # Start with default FPS

    def update_fps(self, fps):
        """
        Updates internal fps (Computed externally in Node based on Images timestamps).
        """

        self.img_avrg_fps = fps

    def is_in_tl_frequency_range(self, point_frequency):
        """
        Verify if the point we have is in the TL green frequency range.
        """
    
        min_freq = self.params["~tl_green_blink_freq"] - self.params["~tl_tolerance_freq_diff"]
        max_freq = self.params["~tl_green_blink_freq"] + self.params["~tl_tolerance_freq_diff"]
        if point_frequency >= min_freq and point_frequency <= max_freq:
            return True
        return False

    def push_camera_image(self, new_img):
        """
        Accumulate the camera images into the buffer
        """

        # Only post image when state is Sensing. No need in other states
        if self.tl_state == TrafficLightState.Sensing:
            # TODO perhaps it is better to just crop the image when received instead of analyzing the full image
            if new_img is not None:
                reduced_img = new_img[:self.params["~max_traffic_light_height"]+1,:].copy()
                with self.buffer_lock:
                    self.buffer.push_frame(reduced_img)

    def reset(self):
        """
        Resets the solver to it's initial state. Should be called when ever the solving should be restarted
        for example when the intersection type is received/change
        """

        print("TrafficLightSolver->reset()")
        with self.buffer_lock:
            self.buffer = PointBuffer(buffer_size=self.params["~tl_buffer_length"],
                                      forget_time=self.params["~tl_buffer_forget_time"],
                                      brightness_threshold=self.params["~tl_brightness_threshold"],
                                      max_diff_threshold=self.params["~tl_max_diff_threshold"],
                                      gblur_sigmaX=self.params["~tl_g_blur_sigma_x"],
                                      gblur_sigmaY=self.params["~tl_g_blur_sigma_y"])
        self.update_tl_state(TrafficLightState.Sensing)

    def should_go(self):
        # TODO other checks?
        return self.tl_state == TrafficLightState.Green

    def update_tl_state(self, new_tl_state):
        """
        Update Traffic Light state
        All state updates should be done here
        """

        # Verify if sate has changed
        if self.tl_state != new_tl_state:
            print(f"TrafficLightSolver: Transition from {self.tl_state} to {new_tl_state}")
            self.tl_state = new_tl_state

    def step_solver(self):
        """
        Step call of the solver to run the analysis on the accumulated images from the camera
        This method is to be called by the Node run() method continuously
        """
    
        # Once we detect at least 1 "light" point flashing we assume it is the traffic light and 
        # traffic light is still "red"
        if len(self.buffer.points) > 0 and self.tl_state == TrafficLightState.Sensing:
            # Check if the flashing frequency is close enough to the TL (7.8Hz)
            # We go through all the detected points. Since we might have captured others from background
            with self.buffer_lock:
                for tl_led_point in self.buffer.points:

                    # TODO Based on some testing with the bots, when the computed FPS (self.img_avrg_fps) is > 30 we get good accurate
                    # results to the real frequencies passing the self.img_avrg_fps to point.get_frequency(). However, for some strange reason
                    # when the computed FPS is way to low (like 18 FPS) both methods (default 30 vs computed fps) are not accurate but using
                    # default 30 FPS get's closer results to the true frequencies.
                    # -> Use max (self.img_avrg_fps, 30).
                    fps = max(self.img_avrg_fps, self.params["~default_img_fps"])

                    if (tl_led_point.coords[0] <= self.params["~max_traffic_light_height"] # coord starts from top of image
                        and (self.is_in_tl_frequency_range(tl_led_point.get_frequency(fps)[0]) # test both fps and self.img_avrg_fps
                        or self.is_in_tl_frequency_range(tl_led_point.get_frequency(self.img_avrg_fps)[0]))):
                        print(f"GREEN Freq : {tl_led_point.get_frequency(fps)[0]}, using avg fps {tl_led_point.get_frequency(self.img_avrg_fps)[0]}, IMG AVG FPS: {self.img_avrg_fps}")
                        self.update_tl_state(TrafficLightState.Green)
                        break # Break right away, we just detected the TL Green state
                    elif tl_led_point.coords[0] <= self.params["~max_traffic_light_height"]:
                        print(f"STILL RED Freq : {tl_led_point.get_frequency(fps)[0]}, using avg fps {tl_led_point.get_frequency(self.img_avrg_fps)[0]}, IMG AVG FPS: {self.img_avrg_fps}")

        # Return the tl green status
        return self.should_go()
