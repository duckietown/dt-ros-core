#!/usr/bin/env python3

# ROS imports
# UKF imports
# The matplotlib imports and the matplotlib.use('Pdf') line make it so that the
# UKF code that imports matplotlib does not complain. Essentially, the
# use('Pdf') call allows a plot to be created without a window (allows it to run
# through ssh)
import rospy
import tf

# Other imports
import numpy as np
import tf.transformations
from filterpy.kalman import MerweScaledSigmaPoints, UnscentedKalmanFilter
from geometry_msgs.msg import PoseStamped, TwistStamped
from sensor_msgs.msg import Imu, Range
from ..state_estimator_abs import StateEstimatorAbs


class UKFStateEstimator7D(StateEstimatorAbs):
    """
    UKF-based state estimator for a 7D state space using various sensor inputs.
    """

    def __init__(self, loop_hz=30.0, altitude_throttled=False, imu_throttled=False, 
                 optical_flow_throttled=False, camera_pose_throttled=False):
        # Call the parent constructor
        super().__init__(loop_hz, state_topic='state_estimator_node/ukf_7d')

        self.ready_to_filter = False
        self.got_altitude = False
        self.got_imu = False
        self.angular_velocity = None

        # Initialize topic names with optional throttling
        self.altitude_topic_str = 'altitude_node' + ('_throttle' if altitude_throttled else '')
        self.imu_topic_str = 'imu' + ('_throttle' if imu_throttled else '')
        self.optical_flow_topic_str = 'camera_node/twist' + ('_throttle' if optical_flow_throttled else '')
        self.camera_pose_topic_str = 'camera_node/pose' + ('_throttle' if camera_pose_throttled else '')

        self.in_callback = False
        self.initialize_estimator()

        # ROS Subscribers
        self._imu_sub = rospy.Subscriber(self.imu_topic_str, Imu, self.imu_data_callback, queue_size=1)
        self._altitude_sub = rospy.Subscriber(self.altitude_topic_str, Range, self.altitude_data_callback, queue_size=1)
        self._of_sub = rospy.Subscriber(self.optical_flow_topic_str, TwistStamped, self.optical_flow_data_callback)
        self._cam_pose_sub = rospy.Subscriber(self.camera_pose_topic_str, PoseStamped, self.camera_pose_data_callback)

    def initialize_estimator(self):
        """ Initialize the UKF estimator parameters. """
        self.state_vector_dim = 7
        self.measurement_vector_dim = 6

        sigma_points = MerweScaledSigmaPoints(n=self.state_vector_dim,
                                              alpha=0.1, beta=2.0,
                                              kappa=(3.0 - self.state_vector_dim))

        self.ukf = UnscentedKalmanFilter(dim_x=self.state_vector_dim,
                                         dim_z=self.measurement_vector_dim,
                                         dt=1.0,
                                         hx=self.measurement_function,
                                         fx=self.state_transition_function,
                                         points=sigma_points)

        self.initialize_ukf_matrices()

    def initialize_ukf_matrices(self):
        """ Initialize the covariance matrices for the UKF. """
        self.ukf.P = np.diag([0.1, 0.1, 0.1, 0.2, 0.2, 0.2, 0.0005])
        self.ukf.Q = np.diag([0.01, 0.01, 0.01, 1.0, 1.0, 1.0, 0.1]) * 0.005
        self.measurement_cov_altitude = np.array([2.2221e-05])
        self.measurement_cov_optical_flow = np.diag([0.01, 0.01])
        self.measurement_cov_camera_pose = np.diag([0.0025, 0.0025, 0.0003])
        self.ukf.R = np.diag([2.2221e-05, 0.0025, 0.0025, 0.01, 0.01, 0.0003])

    def process_pose(self, pose_data : PoseStamped):
        '''Process incoming pose data.'''
        _, _, yaw = tf.transformations.euler_from_quaternion([
            pose_data.pose.orientation.x, pose_data.pose.orientation.y,
            pose_data.pose.orientation.z, pose_data.pose.orientation.w])
        self.last_measurement_vector[1] = pose_data.pose.position.x
        self.last_measurement_vector[2] = pose_data.pose.position.y
        self.last_measurement_vector[5] = yaw

    def process_twist(self, odom : Odometry):
        '''Process incoming twist data.'''
        
        twist = odom.twist

        self.last_measurement_vector[3] = twist.twist.linear.x
        self.last_measurement_vector[4] = twist.twist.linear.y

    def process_range(self, range_data : Range):
        '''Process incoming range data.'''
        self.last_measurement_vector[0] = range_data.range

    def compute_prior(self):
        """ Predict the state using the UKF. """
        self.ukf.predict(dt=self.dt, u=self.last_control_input)
        self.ukf.update(self.last_measurement_vector)

    def imu_data_callback(self, data):
        """ Callback for IMU data. """
        if self.in_callback:
            return
        self.in_callback = True

        self.last_control_input = np.array([data.linear_acceleration.x,
                                            data.linear_acceleration.y,
                                            data.linear_acceleration.z])
        self.angular_velocity = data.angular_velocity

        if not self.ready_to_filter:
            self.initialize_input_time(data)
            self.got_imu = True
            self.ready_to_filter = self.check_if_ready_to_filter()

        self.in_callback = False

    def altitude_data_callback(self, data):
        """ Callback for altitude data. """
        if self.in_callback:
            return
        self.in_callback = True

        if self.ready_to_filter:
            self.update_input_time(data)
        else:
            self.initialize_input_time(data)
            self.ukf.x[2] = data.range
            self.ukf.x[5] = 0.0
            self.ukf.P[2, 2] = self.measurement_cov_altitude[0]
            self.got_altitude = True
            self.ready_to_filter = self.check_if_ready_to_filter()

        self.in_callback = False

    def optical_flow_data_callback(self, data):
        """ Callback for optical flow data. """
        if self.in_callback:
            return
        self.in_callback = True

        if self.ready_to_filter:
            self.update_input_time(data)
        else:
            self.initialize_input_time(data)
            self.ukf.x[3] = data.twist.linear.x
            self.ukf.x[4] = data.twist.linear.y
            self.ukf.P[3, 3] = self.measurement_cov_optical_flow[0, 0]
            self.ukf.P[4, 4] = self.measurement_cov_optical_flow[1, 1]
            self.ready_to_filter = self.check_if_ready_to_filter()

        self.in_callback = False

    def camera_pose_data_callback(self, data):
        """ Callback for camera pose data. """
        if self.in_callback:
            return
        self.in_callback = True

        _, _, yaw = tf.transformations.euler_from_quaternion([
            data.pose.orientation.x, data.pose.orientation.y,
            data.pose.orientation.z, data.pose.orientation.w])

        if self.ready_to_filter:
            self.update_input_time(data)
        else:
            self.initialize_input_time(data)
            self.ukf.x[1] = data.pose.position.x
            self.ukf.x[2] = data.pose.position.y
            self.ukf.x[5] = yaw
            self.ready_to_filter = self.check_if_ready_to_filter()

        self.in_callback = False

    def check_if_ready_to_filter(self):
        """ Check if the estimator is ready to filter. """
        return self.got_altitude and self.got_imu

    def state_transition_function(self, state, dt):
        """ Define the state transition function for the UKF. """
        new_state = state.copy()
        new_state[0] += state[3] * dt
        new_state[1] += state[4] * dt
        new_state[2] += state[5] * dt
        return new_state

    def measurement_function(self, state):
        """ Define the measurement function for the UKF. """
        measurement = np.zeros(self.measurement_vector_dim)
        measurement[0] = state[2]
        measurement[1] = state[0]
        measurement[2] = state[1]
        measurement[3] = state[3]
        measurement[4] = state[4]
        measurement[5] = state[5]
        return measurement
     
def check_positive_float_duration(val):
    """
    Function to check that the --loop_hz command-line argument is a positive
    float.
    """
    value = float(val)
    if value <= 0.0:
        raise argparse.ArgumentTypeError('Loop Hz must be positive')
    return value