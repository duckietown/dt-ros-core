from math import cos, pi
from nav_msgs.msg import Odometry

from geometry_msgs.msg import PoseStamped, TwistStamped
import numpy as np

from sensor_msgs.msg import Imu
import tf.transformations
from .state_estimator_abs import StateEstimatorAbs

from filterpy.kalman import UnscentedKalmanFilter
from filterpy.kalman import MerweScaledSigmaPoints

import tf
import rospy

class StateEstimatorUKF12D(StateEstimatorAbs):
    """A class for filtering data using an Unscented Kalman Filter (UKF)"""

    """
    PSEUDOCODE:

    class StateEstimatorUKF12D(StateEstimatorAbs):
        initialize UKF parameters

        function __init__():
            initialize estimator
            initialize variables
            initialize last control input

        function initialize_estimator():
            set up state vector (12D) and measurement vector (7D)
            create UKF object with sigma points

        function _initialize_ukf_matrices():
            initialize state covariance matrix P
            initialize process noise covariance matrix Q
            initialize measurement covariance matrices for different sensors

        function compute_prior():
            compute prior for UKF based on current state and control input

        function process_pose(pose):
            filter pose data using UKF

        function process_twist(twist):
            if ready_to_filter:
                update input time
                compute prior
                update UKF with twist data
                update current state
            else:
                initialize input time
                initialize UKF state with twist data
                update state covariance matrix
                set got_optical_flow to True
                check if ready to filter

        function process_range(range_reading):
            if ready_to_filter:
                update input time
                compute prior
                update UKF with range reading
                update current state
            else:
                initialize input time
                initialize UKF state with range reading
                update state covariance matrix
                set got_altitude to True
                check if ready to filter

        function process_imu(imu_data):
            extract roll, pitch, yaw from IMU data
            if ready_to_filter:
                update input time
                compute prior
                update UKF with roll, pitch, yaw measurements
                update current state
            else:
                initialize input time
                initialize UKF state with IMU data
                update state covariance matrix
                set got_imu to True
                check if ready to filter

        function update_current_state():
            publish current state estimate and covariance

    main:
        create StateEstimatorUKF12D object
        start main loop
    """

    def __init__(self):
        super().__init__()

        # Initialize the estimator
        self.initialize_estimator()

        self.num_bad_updates = 0
        self.ready_to_filter = False
        self.printed_filter_start_notice = False
        self.got_imu = True
        self.got_optical_flow = False
        self.got_altitude = False

        self.in_callback = False

        self.num_complete_altitude = 0
        self.num_complete_imu = 0
        self.num_complete_optical_flow = 0

        # The last time that we received an input and formed a prediction with
        # the state transition function
        self.last_state_transition_time = None

        # Time in seconds between consecutive state transitions, dictated by
        # when the inputs come in
        self.dt = None

        # Initialize the last control input as 0 m/s^2 along each axis in the
        # body frame
        self.last_control_input = np.array([0.0, 0.0, 0.0])

    def initialize_estimator(self):
        """
        Initialize the parameters of the Unscented Kalman Filter (UKF) that is
        used to estimate the state of the drone.


        The state vector consists of the following column vector.
        Note that FilterPy initializes the state vector with zeros.
        [[x],
         [y],
         [z],
         [x_vel],
         [y_vel],
         [z_vel],
         [roll],
         [pitch],
         [yaw],
         [roll_vel],
         [pitch_vel],
         [yaw_vel]]

        The measurement variables consist of the following column vector:
        [[x_vel],
         [y_vel],
         [yaw_vel],
         [slant_range],
         [roll],
         [pitch],
         [yaw]]
        """

        # Number of state variables being tracked
        self.state_vector_dim = 12
        # Number of measurement variables that the drone receives
        self.measurement_vector_dim = 7

        # Function to generate sigma points for the UKF
        # TODO: Modify these sigma point parameters appropriately. Currently
        #       just guesses
        sigma_points = MerweScaledSigmaPoints(
            n=self.state_vector_dim,
            alpha=0.1,
            beta=2.0,
            kappa=(3.0 - self.state_vector_dim),
        )
        # kappa=0.0)
        # Create the UKF object
        # Note that dt will get updated dynamically as sensor data comes in,
        # as will the measurement function, since measurements come in at
        # distinct rates.
        self.ukf = UnscentedKalmanFilter(
            dim_x=self.state_vector_dim,
            dim_z=self.measurement_vector_dim,
            dt=1.0,
            hx=self.measurement_function,
            fx=self.state_transition_function,
            points=sigma_points,
            residual_x=self.residual_x_account_for_angles,
        )
        self._initialize_ukf_matrices()

    def _initialize_ukf_matrices(self):
        """
        Initialize the covariance matrices of the UKF
        """
        # Initialize state covariance matrix P:
        # TODO: Tune these initial values appropriately. Currently these are
        #       just guesses
        self.ukf.P = np.diag(
            [0.1, 0.1, 0.1, 0.2, 0.2, 0.2, 0.05, 0.05, 0.05, 0.1, 0.1, 0.1]
        )

        # Initialize the process noise covariance matrix Q:
        # TODO: Tune appropriately. Currently just a guess
        # To consider: Changing scale factor by too much could lead to the
        # following error:
        #   "numpy.linalg.linalg.LinAlgError: 3-th leading minor not positive
        #    definite"
        self.ukf.Q = (
            np.diag([0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.5, 0.5, 0.5]) * 0.01
        )

        # Initialize the measurement covariance matrix R for each discrete
        # asynchronous measurement input:
        # Using np.diag makes the covariances 0

        # IR slant range variance (m^2), determined experimentally in a static
        # setup with mean range around 0.335 m:
        # TODO: this should be taken from the datasheet
        self.measurement_cov_altitude = np.array([2.2221e-05])
        # TODO: Tune the following variances appropriately. Currently just
        #       guesses
        # Optical flow variances:
        self.measurement_cov_optical_flow = np.diag([0.01, 0.01, 0.01])
        # Roll-Pitch-Yaw variances:
        self.measurement_cov_rpy = np.diag([0.1, 0.1, 0.1])

    def compute_prior(self):
        """
        Compute the prior for the UKF, based on the current state, a control
        input, and a time step.
        """
        self.ukf.predict(dt=self.dt, u=self.last_control_input)

    def process_pose(self, pose):
        """Filter the pose data using a UKF filter"""
        # UKF update step for pose
        pass

    def process_twist(self, odom: Odometry):
        """
        Handle the receipt of a TwistStamped message from optical flow.
        The message includes:
            - x velocity (m/s)
            - y velocity (m/s)
            - yaw velocity (rad/s)

        This method PREDICTS with the most recent control input and UPDATES.
        """
        twist = odom.twist

        if self.in_callback:
            return
        self.in_callback = True
        if self.ready_to_filter:
            self.update_input_time(twist)
            self.compute_prior()

            # Now that a prediction has been formed to bring the current prior
            # state estimate to the same point in time as the measurement,
            # perform a measurement update with x velocity, y velocity, and yaw
            # velocity data in the TwistStamped message
            # TODO: Verify the units of these velocities that are being
            #       published
            measurement_z = np.array(
                [
                    twist.twist.linear.x,  # x velocity
                    twist.twist.linear.y,  # y velocity
                    twist.twist.angular.z,
                ]
            )  # yaw velocity
            # Ensure that we are using subtraction to compute the residual
            # self.ukf.residual_z = np.subtract
            self.ukf.update(
                measurement_z,
                hx=self.measurement_function_optical_flow,
                R=self.measurement_cov_optical_flow,
            )
            self.update_current_state()
        else:
            self.initialize_input_time(twist)
            # Update the initial state vector of the UKF
            self.ukf.x[3] = twist.twist.linear.x  # x velocity
            self.ukf.x[4] = twist.twist.linear.y  # y velocity
            self.ukf.x[11] = twist.twist.angular.z  # yaw velocity
            # Update the state covariance matrix to reflect estimated
            # measurement error. Variance of the measurement -> variance of
            # the corresponding state variable
            self.ukf.P[3, 3] = self.measurement_cov_optical_flow[0, 0]
            self.ukf.P[4, 4] = self.measurement_cov_optical_flow[1, 1]
            self.ukf.P[11, 11] = self.measurement_cov_optical_flow[2, 2]
            self.got_optical_flow = True
            self.check_if_ready_to_filter()
        self.num_complete_optical_flow += 1
        # print '--OPTICAL FLOW:', self.num_complete_optical_flow
        self.in_callback = False

    def process_range(self, range_reading):
        """
        Handle the receipt of a Range message from the IR sensor.

        This method PREDICTS with the most recent control input and UPDATES.
        """
        if self.in_callback:
            return
        self.in_callback = True
        if self.ready_to_filter:
            self.update_input_time(range_reading)
            rospy.logdebug("BEFORE PREDICT Z:", self.ukf.x[2])
            self.compute_prior()

            # Now that a prediction has been formed to bring the current prior
            # state estimate to the same point in time as the measurement,
            # perform a measurement update with the slant range reading
            measurement_z = np.array([range_reading.range])
            # Ensure that we are using subtraction to compute the residual
            # self.ukf.residual_z = np.subtract
            rospy.logdebug("AFTER PREDICT Z:", self.ukf.x[2])
            # Multiply slant range by cos(roll)*cos(pitch) to get altitude estimate
            raw_slant_range_as_altitude = (
                measurement_z[0] * cos(self.ukf.x[6]) * cos(self.ukf.x[7])
            )
            rospy.logdebug(
                f"Raw slant range {measurement_z[0]}transformed to altitude:{ raw_slant_range_as_altitude}"
            )
            self.ukf.update(
                measurement_z,
                hx=self.measurement_function_altitude,
                R=self.measurement_cov_altitude,
            )

            # For testing, don't use the unscented transform for residual computation
            # temp_residual = measurement_z - self.ukf.sigmas_h[0]
            # print 'TEMP RESIDUAL:', temp_residual
            # self.ukf.x = self.ukf.x_prior + np.dot(self.ukf.K, temp_residual)

            rospy.logdebug("AFTER UPDATE Z:", self.ukf.x[2])
            # print 'KALMAN GAIN Z:', self.ukf.K[2]
            # print 'RESIDUAL:', self.ukf.y
            # print
            if not (
                (raw_slant_range_as_altitude <= self.ukf.x[2] <= self.ukf.x_prior[2])
                or (raw_slant_range_as_altitude >= self.ukf.x[2] >= self.ukf.x_prior[2])
            ):
                self.num_bad_updates += 1
                # print self.num_bad_updates, 'BAD UPDATE...\n\n\n\n\n\n\n'
            self.update_current_state()
        else:
            self.initialize_input_time(range_reading)
            # Got a raw slant range reading, so update the initial state
            # vector of the UKF
            self.ukf.x[2] = range_reading.range
            # Update the state covariance matrix to reflect estimated
            # measurement error. Variance of the measurement -> variance of
            # the corresponding state variable
            self.ukf.P[2, 2] = self.measurement_cov_altitude[0]
            self.got_altitude = True
            self.check_if_ready_to_filter()
        self.num_complete_altitude += 1
        # print '--IR:', self.num_complete_altitude
        # print
        self.in_callback = False

    def process_imu(self, imu_data: Imu):
        """
        Handle the receipt of an Imu message, which includes linear
        accelerations (m/s^2) to be treated as a control input in the UKF and
        orientation to be treated as measurement inputs.

        This method PREDICTS with a control input and UPDATES.
        """
        if self.in_callback:
            return
        self.in_callback = True
        euler_angles = tf.transformations.euler_from_quaternion(
            [
                imu_data.orientation.x,
                imu_data.orientation.y,
                imu_data.orientation.z,
                imu_data.orientation.w,
            ]
        )
        roll = euler_angles[0]
        pitch = euler_angles[1]
        yaw = euler_angles[2]
        self.last_control_input = np.array(
            [
                imu_data.linear_acceleration.x,
                imu_data.linear_acceleration.y,
                imu_data.linear_acceleration.z,
            ]
        )
        if self.ready_to_filter:
            self.update_input_time(imu_data)
            self.compute_prior()

            # Now that a prediction has been formed, perform a measurement
            # update with the roll-pitch-yaw data in the Imu message
            measurement_z = np.array([roll, pitch, yaw])

            # Ensure that we are computing the residual for angles
            # self.ukf.residual_z = self.angle_residual
            # Actually, looks like euler_from_quaternion can return negative
            # angles in radians
            # self.ukf.residual_z = np.subtract
            # TODO: Look into the range of angles returned by
            #       euler_from_quaternion. As there are negatives, it would seem
            #       possible that the range be between -pi and pi radians...
            self.ukf.update(
                measurement_z,
                hx=self.measurement_function_rpy,
                R=self.measurement_cov_rpy,
            )
            self.update_current_state()
        else:
            self.initialize_input_time(imu_data)
            # Update the initial state vector of the UKF
            self.ukf.x[6] = roll
            self.ukf.x[7] = pitch
            self.ukf.x[8] = yaw
            # Update the state covariance matrix to reflect estimated
            # measurement error. Variance of the measurement -> variance of
            # the corresponding state variable
            self.ukf.P[6, 6] = self.measurement_cov_rpy[0, 0]
            self.ukf.P[7, 7] = self.measurement_cov_rpy[1, 1]
            self.ukf.P[8, 8] = self.measurement_cov_rpy[2, 2]
            self.got_imu = True
            self.check_if_ready_to_filter()
        self.num_complete_imu += 1
        self.in_callback = False

    def update_current_state(self):
        """
        Publish the current state estimate and covariance from the UKF. This is
        a State message containing:
            - Header
            - PoseWithCovariance
            - TwistWithCovariance
        """
        state_msg = Odometry()
        state_msg.header.stamp.secs = self.last_time_secs
        state_msg.header.stamp.nsecs = self.last_time_nsecs
        state_msg.header.frame_id = "global"

        quaternion = self.get_quaternion_from_ukf_rpy()

        # Get the current state estimate from self.ukf.x
        state_msg.pose.pose.position.x = self.ukf.x[0]
        state_msg.pose.pose.position.y = self.ukf.x[1]
        state_msg.pose.pose.position.z = self.ukf.x[2]
        state_msg.twist.twist.linear.x = self.ukf.x[3]
        state_msg.twist.twist.linear.y = self.ukf.x[4]
        state_msg.twist.twist.linear.z = self.ukf.x[5]
        state_msg.pose.pose.orientation.x = quaternion[0]
        state_msg.pose.pose.orientation.y = quaternion[1]
        state_msg.pose.pose.orientation.z = quaternion[2]
        state_msg.pose.pose.orientation.w = quaternion[3]

        # TODO: Look into RPY velocities versus angular velocities about x, y, z axes?
        # For the time being, using Euler rates:
        state_msg.twist.twist.angular.x = self.ukf.x[9]  # roll rate
        state_msg.twist.twist.angular.y = self.ukf.x[10]  # pitch rate
        state_msg.twist.twist.angular.z = self.ukf.x[11]  # yaw rate

        # Extract the relevant covariances from self.ukf.P, make into 36-element
        # arrays, in row-major order, according to ROS msg docs
        P = self.ukf.P
        state_msg.pose.covariance = np.concatenate(
            (
                P[0, 0:3],
                P[0, 6:9],
                P[1, 0:3],
                P[1, 6:9],
                P[2, 0:3],
                P[2, 6:9],
                P[6, 0:3],
                P[6, 6:9],
                P[7, 0:3],
                P[7, 6:9],
                P[8, 0:3],
                P[8, 6:9],
            ),
            axis=0,
        )
        state_msg.twist.covariance = np.concatenate(
            (
                P[3, 3:6],
                P[3, 9:12],
                P[4, 3:6],
                P[4, 9:12],
                P[5, 3:6],
                P[5, 9:12],
                P[9, 3:6],
                P[9, 9:12],
                P[10, 3:6],
                P[10, 9:12],
                P[11, 3:6],
                P[11, 9:12],
            ),
            axis=0,
        )

        self.state = state_msg

    def check_if_ready_to_filter(self):
        self.ready_to_filter = (
            self.got_imu and self.got_optical_flow and self.got_altitude
        )

    def get_quaternion_from_ukf_rpy(self):
        # TODO: Should we use the raw roll, pitch, and yaw values that come in
        #       at the same time step as the linear accelerations?
        return tf.transformations.quaternion_from_euler(
            self.ukf.x[6], self.ukf.x[7], self.ukf.x[8]
        )

    def initialize_input_time(self, msg):
        """
        Initialize the input time (self.last_state_transition_time) based on the
        timestamp in the header of a ROS message. This is called before we start
        filtering in order to attain an initial time value, which enables us to
        then compute a time interval self.dt by calling self.update_input_time()

        msg : a ROS message that includes a header with a timestamp
        """
        self.last_time_secs = msg.header.stamp.secs
        self.last_time_nsecs = msg.header.stamp.nsecs
        self.last_state_transition_time = (
            self.last_time_secs + self.last_time_nsecs * 1e-9
        )

    def update_input_time(self, msg):
        """
        Update the time at which we have received the most recent input, based
        on the timestamp in the header of a ROS message

        msg : a ROS message that includes a header with a timestamp that
              indicates the time at which the respective input was originally
              recorded
        """
        self.last_time_secs = msg.header.stamp.secs
        self.last_time_nsecs = msg.header.stamp.nsecs
        new_time = self.last_time_secs + self.last_time_nsecs * 1e-9
        # Compute the time interval since the last state transition / input
        self.dt = new_time - self.last_state_transition_time
        # Set the current time at which we just received an input
        # to be the last input time
        self.last_state_transition_time = new_time

    def apply_quaternion_vector_rotation(self, original_vector):
        """
        Rotate a vector from the drone's body frame to the global frame using
        quaternion-vector multiplication
        """
        # Use quaternion-vector multiplication instead of a rotation matrix to
        # rotate the vector.
        # This quaternion describes rotation from the global frame to the body
        # frame, so take the quaternion's inverse by negating its real
        # component (w)
        quat_global_to_body = self.get_quaternion_from_ukf_rpy()
        quat_body_to_global = list(quat_global_to_body)  # copy the quaternion
        quat_body_to_global[3] = -quat_body_to_global[3]
        original_vector_as_quat = list(original_vector)
        original_vector_as_quat.append(0.0)  # vector as quaternion with w=0
        # Apply quaternion rotation on a vector: q*v*q', where q is the rotation
        # quaternion, v is the vector (a "pure" quaternion with w=0), and q' is
        # the conjugate of the quaternion q
        original_vector_rotated = tf.transformations.quaternion_multiply(
            tf.transformations.quaternion_multiply(
                quat_body_to_global, original_vector_as_quat
            ),
            tf.transformations.quaternion_conjugate(quat_body_to_global),
        )
        # Drop the real part w=0
        original_vector_rotated = original_vector_rotated[:3]
        return original_vector_rotated

    def state_transition_function(self, x, dt, u):
        """
        The state transition function to compute the prior in the prediction
        step, propagating the state to the next time step.

        x : current state. A NumPy array
        dt : time step. A float
        u : control input. A NumPy array
        """
        F = np.eye(self.state_vector_dim)
        F[0, 3] = dt
        F[1, 4] = dt
        F[2, 5] = dt
        F[6, 9] = dt
        F[7, 10] = dt
        F[8, 11] = dt

        # Compute the change from the control input
        accelerations_global_frame = self.apply_quaternion_vector_rotation(u)
        velocities_global_frame = accelerations_global_frame * dt
        change_from_control_input = np.array(
            [
                0,
                0,
                0,
                velocities_global_frame[0],
                velocities_global_frame[1],
                velocities_global_frame[2],
                0,
                0,
                0,
                0,
                0,
                0,
            ]
        )
        x_output = np.dot(F, x) + change_from_control_input
        # x_output = self.correct_fringe_angles(x_output)
        return x_output

    def residual_x_account_for_angles(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        Computes the residual for the state vector x, accounting for angles.

        Args:
            a: A NumPy array with shape (12,), representing the state vector.
            b: A NumPy array with shape (12,), representing the state vector.

        Returns:
            A NumPy array with shape (12,), representing the residual for each state variable.
        """

        output_residual = np.zeros(12)
        output_residual[:6] = a[:6] - b[:6]

        angles = a[6:9] - b[6:9]
        angles[angles > pi] -= 2 * pi
        angles[angles < -pi] += 2 * pi
        output_residual[6:9] = angles

        output_residual[9:] = a[9:] - b[9:]

        return output_residual

    def correct_fringe_angles(self, x_in_transition):
        """
        Check if the state transition involves fringe angles, i.e., if an
        orientation angle transitions from 359 degrees to 362 degrees, then the
        new angle should read 2 degrees. Likewise, a transition from 2 degrees
        to -3 degrees should read 357 degrees. (Examples given in degrees, but
        in the implementation angles are given in radians) This is just a matter
        of applying the modulo operator to each transitioned angle.
        """
        x_in_transition[6] = x_in_transition[6] % 2 * pi  # roll angle
        x_in_transition[7] = x_in_transition[7] % 2 * pi  # pitch angle
        x_in_transition[8] = x_in_transition[8] % 2 * pi  # yaw angle
        return x_in_transition

    def measurement_function(self, x: np.ndarray) -> np.ndarray:
        """
        The "complete" measurement function if the measurement vector z were to
        be comprised of all measurement variables at any given timestep. This
        function does not actually get called, since measurements come in at
        distinct rates, but it is passed into the UKF object as the default
        measurement function.

        x : current state. A NumPy array
        """
        # Roll and pitch values from the prior state estimate
        phi = x[6]  # roll in radians
        theta = x[7]  # pitch in radians
        # Conversion from altitude (alt) to slant range (r)
        alt_to_r = 1 / (cos(theta) * cos(phi))
        H = np.array(
            [
                [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                [0, 0, alt_to_r, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
            ]
        )
        hx_output = np.dot(H, x)
        return hx_output

    def measurement_function_altitude(self, x):
        """
        For use when the measurement vector z is just the slant range reading
        from the IR sensor

        x : current state. A NumPy array
        """
        # Roll and pitch values from the prior state estimate
        phi = x[6]  # roll in radians
        theta = x[7]  # pitch in radians
        # Conversion from altitude (alt) to slant range (r)
        alt_to_r = 1 / (cos(theta) * cos(phi))
        H = np.array([[0, 0, alt_to_r, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
        hx_output = np.dot(H, x)
        # print 'hx_output:', hx_output
        return hx_output

    def measurement_function_optical_flow(self, x):
        """
        For use when the measurement vector z is comprised of x-velocity,
        y-velocity, and yaw velocity from the camera's optical flow

        x : current state. A NumPy array
        """
        H = np.array(
            [
                [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            ]
        )
        hx_output = np.dot(H, x)
        return hx_output

    def measurement_function_rpy(self, x: np.ndarray) -> np.ndarray:
        """
        For use when the measurement vector z is comprised of roll, pitch, and
        yaw readings from the IMU

        x : current state. A NumPy array
        """
        H = np.array(
            [
                [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
            ]
        )
        hx_output = np.dot(H, x)
        return hx_output
