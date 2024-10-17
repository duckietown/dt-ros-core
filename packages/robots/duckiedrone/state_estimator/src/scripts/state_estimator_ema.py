#!/usr/bin/env python3

from math import cos
from sensor_msgs.msg import Imu, Range
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from .state_estimator_abs import StateEstimatorAbs


class StateEstimatorEMA(StateEstimatorAbs):
    ''' A class for filtering data using an Exponential Moving Average (EMA) '''

    def __init__(self):
        super().__init__()

        # TODO: These should be params
        self.alpha_pose = 0.2  # Smoothing factor for pose
        self.alpha_twist = 0.4  # Smoothing factor for twist
        self.alpha_range = 0.8  # Smoothing factor for range
        self.max_range = 5.0    # Example maximum range, should be set appropriately

        # Initialize the estimator
        self.initialize_estimator()

    def initialize_estimator(self):
        """ Initialize the EMA estimator parameters. """
        # Any additional initialization can be added here
        pass

    def compute_prior(self):
        """ Predict the state in the context of EMA filtering (if needed). """
        # EMA does not involve a prediction step like UKF, so this can be left empty or handle periodic updates
        pass

    def process_pose(self, pose : PoseStamped):
        """ Filter the pose data using an EMA filter """
        last_position = self.state.pose.pose.position
        position_reading = pose.pose.position

        smoothed_x = (1.0 - self.alpha_pose) * last_position.x + self.alpha_pose * position_reading.x
        smoothed_y = (1.0 - self.alpha_pose) * last_position.y + self.alpha_pose * position_reading.y

        self.state.pose.pose.position.x = smoothed_x
        self.state.pose.pose.position.y = smoothed_y

    def process_twist(self, odom : Odometry):
        """ Filter the twist data using an EMA filter """
        velocity = self.state.twist.twist.linear
        new_vel = odom.twist.twist.linear

        self.calc_angle_comp_values()  # Assuming this is used to correct the measurements

        velocity.x = self.near_zero((1.0 - self.alpha_twist) * velocity.x + self.alpha_twist * (new_vel.x - self.mw_angle_comp_x))
        velocity.y = self.near_zero((1.0 - self.alpha_twist) * velocity.y + self.alpha_twist * (new_vel.y - self.mw_angle_comp_y))

        self.state.twist.twist.linear = velocity

    def process_range(self, range_reading : Range):
        """ Filter the range data using an EMA filter """
        r, p, _ = self.get_r_p_y()
        curr_altitude = range_reading.range * cos(r) * cos(p)
        prev_altitude = self.state.pose.pose.position.z

        smoothed_altitude = (1.0 - self.alpha_range) * curr_altitude + self.alpha_range * prev_altitude
        smoothed_altitude = max(0, min(smoothed_altitude, self.max_range * 0.8))

        self.state.pose.pose.position.z = smoothed_altitude

    def calc_angle_comp_values(self):
        # TODO: implement this method
        """ Calculate angle compensation values (dummy implementation). """
        # This method would need to be fully implemented based on your specific requirements
        self.mw_angle_comp_x = 0.0
        self.mw_angle_comp_y = 0.0

    @staticmethod
    def near_zero(value, epsilon=1e-6):
        """ Helper function to zero out small values for numerical stability. """
        return value if abs(value) > epsilon else 0.0

    @staticmethod
    def get_r_p_y():
        # TODO: implement this method so that we can drop the altitude node
        """ Extract roll, pitch, yaw from the current state (dummy implementation). """
        # This method would need to be implemented to extract the correct roll, pitch, and yaw values
        return 0.0, 0.0, 0.0

    def process_imu(self, imu_data: Imu):
        return super().process_imu(imu_data)