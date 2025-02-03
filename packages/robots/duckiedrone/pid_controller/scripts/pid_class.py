#!/usr/bin/env python3

from cmath import inf
import logging
from typing import List, Optional, Tuple

import numpy as np
from simple_pid import PID as SimplePID
from three_dim_vec import Error
from duckietown_msgs.msg import PIDState

class PIDaxis:

    def __init__(self,
                 kp, ki, kd,
                 i_range=(1000,2000),
                 d_range=None,
                 control_range : Tuple[int] = (1000, 2000),
                 midpoint=1500,
                 smoothing=True):
        
        self.kp = kp
        self.ki = ki
        self.kd = kd
        
        self.i_range = i_range
        self.d_range = d_range
        self.control_range = control_range
        assert control_range[0] < midpoint < control_range[1]
    
        self.midpoint = midpoint
        self.smoothing = smoothing
        
        self.init_i = 0.0

        # Internal variables
        self._old_err = None
        self._p = 0
        self.integral = self.init_i
        
        # effective only once
        self.init_i = 0.0
        self._d = 0
        self._dd = 0
        self._ddd = 0
        
    def reset(self):
        self._old_err = None
        self._p = 0
        self.integral = self.init_i
        # effective only once
        self.init_i = 0.0
        self._d = 0
        self._dd = 0
        self._ddd = 0

    def step(self, err, delta_t) -> float:
        if delta_t <= 0:
            return 0

        if self._old_err is None:
            # First time around prevent d term spike
            self._old_err = err

        # Compute the p component
        self._p = err * self.kp

        # Compute the i component
        self.integral += err * self.ki * delta_t
        if self.i_range is not None:
            self.integral = np.clip(self.integral, self.i_range[0], self.i_range[1])

        # Compute the d component
        self._d = (err - self._old_err) * self.kd / delta_t
        if self.d_range is not None:
            self._d = max(self.d_range[0], min(self._d, self.d_range[1]))
        self._old_err = err

        # Smooth over the last three d terms
        if self.smoothing:
            self._d = (self._d * 8.0 + self._dd * 5.0 + self._ddd * 2.0) / 15.0
            self._ddd = self._dd
            self._dd = self._d

        # Calculate control output
        raw_output = self._p + self.integral + self._d
        output = np.clip(raw_output + self.midpoint, self.control_range[0], self.control_range[1])

        return output


# noinspection DuplicatedCode
class PID:
    height_factor = 1.238
    battery_factor = 0.75
    PID_SAMPLE_RATE = 50
    TARGET_ALTITUDE = 1.0

    def __init__(
        self,
        roll=PIDaxis(
            4.0,
            1.0,
            0.0,
            control_range=(1400, 1600),
            midpoint=1500,
            i_range=(-100, 100),
        ),
        roll_low=PIDaxis(
            0.0,
            0.5,
            0.0,
            control_range=(1400, 1600),
            midpoint=1500,
            i_range=(-150, 150),
        ),
        pitch=PIDaxis(
            4.0,
            1.0,
            0.0,
            control_range=(1400, 1600),
            midpoint=1500,
            i_range=(-100, 100),
        ),
        pitch_low=PIDaxis(
            0.0,
            0.5,
            0.0,
            control_range=(1400, 1600),
            midpoint=1500,
            i_range=(-150, 150),
        ),
        yaw=PIDaxis(0.0, 0.0, 0.0),
        thrust=SimplePID(
            0.10,
            0.05,
            0.04,
            setpoint=TARGET_ALTITUDE,
            sample_time=1 / PID_SAMPLE_RATE,
            output_limits=(0, 1),
            time_fn= lambda: rospy.get_time()
        ),
    ):
        self.trim_controller_cap_plane = 0.05
        self.trim_controller_thresh_plane = 0.0001

        self.roll = roll
        self.roll_low = roll_low

        self.pitch = pitch
        self.pitch_low = pitch_low

        self.yaw = yaw

        self.trim_controller_cap_throttle = 5.0
        self.trim_controller_thresh_throttle = 5.0

        self.thrust = thrust

        self._t = None

        # Tuning values specific to each drone
        # TODO: these should be params
        if self.roll_low is not None:
            self.roll_low.init_i = 0.31
        if self.pitch_low is not None:
            self.pitch_low.init_i = -1.05
        self.reset()

    def reset(self):
        """ Reset each pid and restore the initial i terms """
        # reset time variable
        self._t = None

        # reset individual PIDs
        for pid in [self.roll, self.roll_low, self.pitch, self.pitch_low, self.thrust]:
            if pid is not None:
                pid.reset()

    def step(self, error : Error, t :float, cmd_yaw_velocity=0) -> List:
        """
        Compute the control variables from the error using the step methods
        of each axis PID controller.

        Parameters:
        error (object): An object containing the error values for each axis (x, y, z).
        cmd_yaw_velocity (float, optional): The commanded yaw velocity. Defaults to 0.

        list: A list of control variables [roll, pitch, yaw, thrust].

        Notes:
        - The first time this method is called, it prevents a time spike by setting the elapsed time to 1.
        - The roll and pitch commands are computed using the `compute_axis_command` method.
        - The yaw command is computed by adding the commanded yaw velocity to a base value of 1500.
        - The thrust command is computed using the `thrust` method with the z-axis error and thrust setpoint.
        """

        # First time around prevent time spike
        if self._t is None:
            time_elapsed = 1
        else:
            time_elapsed = t - self._t

        self._t = t

        cmd_roll = self.compute_axis_command(
            error.y,
            time_elapsed,
            pid_low=self.roll_low,
            pid=self.roll,
            trim_controller=self.trim_controller_cap_plane
            )

        cmd_pitch = self.compute_axis_command(
            error.x,
            time_elapsed,
            pid_low=self.pitch_low,
            pid=self.pitch,
            trim_controller=self.trim_controller_cap_plane
            )

        cmd_yaw = 1500 + cmd_yaw_velocity

        cmd_thrust = self.thrust(error.z + self.thrust.setpoint)


        return [
            1500, # TODO: this should return a commanded attitude rather than RC commands!
            1500,
            cmd_yaw,
            cmd_thrust,
        ]

    def compute_axis_command(
        self,
        error: float,
        time_elapsed: float,
        pid: PIDaxis,
        pid_low: Optional[PIDaxis] = None,
        trim_controller: float = 5,
    ) -> float:
        if pid_low is None:
            cmd = pid.step(error, time_elapsed)
            return cmd

        if abs(error) < self.trim_controller_thresh_plane:
            # pass the high rate i term off to the low rate pid
            pid_low.integral += pid.integral
            pid.integral = 0
            # set the roll value to just the output of the low rate pid
            cmd = pid_low.step(error, time_elapsed)
        else:
            if error > trim_controller:
                pid_low.step(trim_controller, time_elapsed)
            elif error < -trim_controller:
                pid_low.step(-trim_controller, time_elapsed)
            else:
                pid_low.step(error, time_elapsed)
            # cmd = pid_low.integral + pid.step(error, time_elapsed)
            cmd = pid.step(error, time_elapsed)
        
        return cmd
