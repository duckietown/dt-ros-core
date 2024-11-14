#!/usr/bin/env python3

import logging
from typing import List, Optional
import rospy


#####################################################
#						PID							#
#####################################################
class PIDaxis:

    def __init__(self,
                 kp, ki, kd,
                 i_range=None,
                 d_range=None,
                 control_range=(1000, 2000),
                 midpoint=1500,
                 smoothing=True):
        # Tuning
        self.kp = kp
        self.ki = ki
        self.kd = kd
        # Config
        self.i_range = i_range
        self.d_range = d_range
        self.control_range = control_range
        self.midpoint = midpoint
        self.smoothing = smoothing
        # initial i value
        self.init_i = 0.0

        # Internal
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

    def step(self, err, time_elapsed) -> int:
        if time_elapsed == 0:
            return 0

        if self._old_err is None:
            # First time around prevent d term spike
            self._old_err = err

        # Find the p component
        self._p = err * self.kp

        # Find the i component
        self.integral += err * self.ki * time_elapsed
        if self.i_range is not None:
            self.integral = max(self.i_range[0], min(self.integral, self.i_range[1]))

        # Find the d component
        self._d = (err - self._old_err) * self.kd / time_elapsed
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
        output = min(max(raw_output + self.midpoint, self.control_range[0]), self.control_range[1])

        return int(output)


# noinspection DuplicatedCode
class PID:
    height_factor = 1.238
    battery_factor = 0.75

    def __init__(self,
                 roll=PIDaxis(
                     4.0, 1.0, 0.0,
                     control_range=(1400, 1600),
                     midpoint=1500,
                     i_range=(-100, 100)
                 ),
                 roll_low=PIDaxis(
                     0.0, 0.5, 0.0,
                     control_range=(1400, 1600),
                     midpoint=1500,
                     i_range=(-150, 150)
                 ),

                 pitch=PIDaxis(
                     4.0, 1.0, 0.0,
                     control_range=(1400, 1600),
                     midpoint=1500,
                     i_range=(-100, 100)
                 ),
                 pitch_low=PIDaxis(
                     0.0, 0.5, 0.0,
                     control_range=(1400, 1600),
                     midpoint=1500,
                     i_range=(-150, 150)
                 ),

                 yaw=PIDaxis(0.0, 0.0, 0.0),

                 # Kv 2300 motors have midpoint 1300, Kv 2550 motors have midpoint 1250
                 # height_safety_here (in the sense that the motors are limited)
                 # 1.0, 0.5, 2.0
                 # 1.0, 0.05, 2.0
                 throttle=PIDaxis(
                     5.0 / height_factor * battery_factor,
                     0.5 / height_factor * battery_factor,
                     10.0 / height_factor * battery_factor,
                     i_range=(-400, 400),
                     control_range=(1200, 1500),
                     d_range=(-40, 40),
                     midpoint=1500
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

        self.throttle = throttle

        self._t = None

        # Tuning values specific to each drone
        # TODO: these should be params
        self.roll_low.init_i = 0.31
        self.pitch_low.init_i = -1.05
        self.reset()

    def reset(self):
        """ Reset each pid and restore the initial i terms """
        # reset time variable
        self._t = None

        # reset individual PIDs
        self.roll.reset()
        self.roll_low.reset()
        self.pitch.reset()
        self.pitch_low.reset()
        self.throttle.reset()

    def step(self, error, cmd_yaw_velocity=0) -> List[int]:
        """ Compute the control variables from the error using the step methods
        of each axis pid.
        """
        # First time around prevent time spike
        if self._t is None:
            time_elapsed = 1
        else:
            time_elapsed = rospy.get_time() - self._t

        self._t = rospy.get_time()

        # Compute roll command
        ######################
        # if the x velocity error is within the threshold
        cmd_r = self.compute_axis_command(
            error.x,
            time_elapsed,
            pid_low=self.roll_low,
            pid=self.roll,
            trim_controller=self.trim_controller_cap_plane
            )

        # Compute pitch command
        cmd_p = self.compute_axis_command(
            error.y,
            time_elapsed,
            pid_low=self.pitch_low,
            pid=self.pitch,
            trim_controller=self.trim_controller_cap_plane
            )
        # Compute yaw command
        cmd_y = 1500 + cmd_yaw_velocity

        cmd_t = self.compute_axis_command(
            error.z,
            time_elapsed,
            pid_low=None,
            pid=self.throttle,
            trim_controller=self.trim_controller_cap_throttle
        )
        return [cmd_r, cmd_p, cmd_y, cmd_t]

    def compute_axis_command(
        self,
        error: float,
        time_elapsed: float,
        pid: PIDaxis,
        pid_low: Optional[PIDaxis] = None,
        trim_controller: float = 5,
    ) -> int:
        if pid_low is None:
            cmd_r = pid.step(error, time_elapsed)
            return cmd_r

        if abs(error) < self.trim_controller_thresh_plane:
            # pass the high rate i term off to the low rate pid
            pid_low.integral += pid.integral
            pid.integral = 0
            # set the roll value to just the output of the low rate pid
            cmd_r = pid_low.step(error, time_elapsed)
        else:
            if error > trim_controller:
                pid_low.step(trim_controller, time_elapsed)
            elif error < -trim_controller:
                pid_low.step(-trim_controller, time_elapsed)
            else:
                pid_low.step(error, time_elapsed)
            cmd_r = pid_low.integral + pid.step(error, time_elapsed)
        
        return cmd_r
