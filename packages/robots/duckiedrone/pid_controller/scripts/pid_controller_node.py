#!/usr/bin/env python3

from enum import IntEnum
from math import inf
from typing import List
import rospy
import tf
from duckietown_msgs.msg import DroneControl as RC, PIDDiagnostics as PIDDebugInfo, PIDState
from mavros_msgs.msg import State as FCUState
from mavros_msgs.srv import CommandBool, SetMode
from geometry_msgs.msg import Pose, Twist
from nav_msgs.msg import Odometry
from std_msgs.msg import Empty, Bool, Float32
from std_srvs.srv import SetBool, SetBoolResponse, SetBoolRequest

from duckietown.dtros import DTROS, DTParam, NodeType, ParamType
from pid_class import PID as dtPID, PIDaxis
from three_dim_vec import Position, Velocity, Error, RPY

class DroneMode(IntEnum):
    DISARMED = 0
    ARMED = 1
    FLYING = 2

class PIDControllerNode(DTROS):
    """
    Controls the flight of the drone by running a PID controller on the
    error calculated by the desired and current velocity and position of the drone.

    The parameters of the PID controller are defined in the DTROS configuration file.
    They can be tuned on the fly to adjust the drone's response by changing the value of 
    the respective rosparam parameter using `rosparam set`
    (e.g. on the 'endeavour' drone the proportional gain of the pitch PID can be set to 1.0 by
    calling `rosparam set /endeavour/pid_controller_node/pitch_kp 1.0').
    """

    def __init__(self):
        super(PIDControllerNode, self).__init__(node_name="pid_controller_node",
                                            node_type=NodeType.CONTROL)
        
        self.frequency = rospy.get_param("~frequency")
        self.max_height = rospy.get_param("~max_height")
        self.hover_height = rospy.get_param("~hover_height")

        self.previous_mode = DroneMode.DISARMED
        self.current_mode = DroneMode.DISARMED

        # Initialize in velocity control
        self.position_control = False
        self.last_position_control = False

        self.current_position = Position()
        self.desired_position = Position(z=self.hover_height)
        self.last_desired_position = self.desired_position

        self.current_velocity = Velocity()
        self.desired_velocity = Velocity()

        self.position_error = Error()
        self.velocity_error = Error()

        # Set the distance that a velocity command will move the drone (m)
        self.desired_velocity_travel_distance = 0.1

        # Set a static duration that a velocity command will be held
        self.desired_velocity_travel_time = 0.1

        # Set a static duration that a yaw velocity command will be held
        self.desired_yaw_velocity_travel_time = 0.25

        # Store the start time of the desired velocities
        self.desired_velocity_start_time = None
        self.desired_yaw_velocity_start_time = None

        # Initialize the primary PID
        self.pid = dtPID()
        self.pid.thrust.setpoint = self.hover_height

        # Initialize the error used for the PID which is vx, vy, z where vx and
        # vy are velocities, and z is the error in the altitude of the drone
        self.pid = PID(
                roll=PIDaxis(
                    0.5, 0.0, 0.0,
                    control_range=(-inf, +inf),
                    midpoint=0,
                    i_range=(-100, 100)
                 ),
                 roll_low=None,
                 pitch=PIDaxis(
                    0.5, 0.0, 0.0,
                    control_range=(-inf, +inf),
                    midpoint=0,
                    i_range=(-100, 100)
                 ),
                 pitch_low=None,
        )

        self.initialize_pid_gains()
        self.pid_error = Error()

        # Initialize the 'position error to velocity error' PIDs:
        # TODO: read the PID gains from a config file and allow them to be set via parameter server or a service
        # left/right (roll) pid
        self.lr_pid = PIDaxis(kp=20.0, ki=5.0, kd=10.0, midpoint=0, control_range=(-10.0, 10.0))
        # front/back (pitch) pid
        self.fb_pid = PIDaxis(kp=20.0, ki=5.0, kd=10.0, midpoint=0, control_range=(-10.0, 10.0))

        # Initialize the pose callback time
        self.last_pose_time = None

        self.desired_yaw_velocity = 0

        self.current_rpy = RPY()
        self.previous_rpy = RPY()

        self.current_state = Odometry()
        self.previous_state = Odometry()

        # Used to determine if the drone is moving between desired positions
        self.moving = False

        # Determines the maximum acceptable position error magnitude, an error
        # greater than this value will overide the drone into velocity control
        self.safety_threshold = 1.5

        # determines if the position of the drone is known
        self.lost = False

        # determines if the desired poses are aboslute or relative to the drone
        self.absolute_desired_position = False

        # TODO: refactor path_planning into a service
        # determines whether to use open loop velocity path planning which is
        # accomplished by calculate_travel_time
        self.path_planning = True

        # publishers
        self.cmd_pub = rospy.Publisher(
            "~commands",
            AttitudeTarget,
            queue_size=1
        )
        self.debug_pub = rospy.Publisher("~debug/pid", PIDDebugInfo, queue_size=10)

        self.position_control_pub = rospy.Publisher(
            "~position_control",
            Bool,
            queue_size=1
        )
        self.heartbeat_pub = rospy.Publisher(
            "~heartbeat",
            Empty,
            queue_size=1
        )
        self._desired_height_pub = rospy.Publisher(
            "~desired/height",
            Float32,
            queue_size=1,
            latch=True
        )

        rospy.Subscriber("~mode", FCUState, self.current_mode_callback, queue_size=1)
        rospy.Subscriber("~state", Odometry, self.current_state_callback, queue_size=1)

        # TODO: refactor callbacks
        rospy.Subscriber("desired/pose", Pose, self.desired_pose_callback, queue_size=1)
        rospy.Subscriber("desired/twist", Twist, self.desired_twist_callback, queue_size=1)
        rospy.Subscriber("camera_node/lost", Bool, self.lost_callback, queue_size=1)

        # TODO: transform reset_transform and position_control switch into services
        rospy.Subscriber("reset_transform", Empty, self.reset_callback, queue_size=1)
        rospy.Subscriber("~position_control", Bool, self.position_control_callback, queue_size=1)


        # Create a service proxy
        rospy.wait_for_service("~set_mode")
        self.set_mode_service = rospy.ServiceProxy("~set_mode", SetMode)
        self.set_mode(mode="GUIDED_NOGPS")

        rospy.Service("~takeoff", SetBool, self.takeoff_srv)
        rospy.Service("~land", SetBool, self.land_srv)
        
        # publish internal desired pose (hover pose)
        self._desired_height_pub.publish(Float32(self.desired_position.z))

    def _set_pid_gain(self, axis: str, gain: str, value: float):
        """
        Sets the specified gain for the specified axis in the PID controller.

        Args:
            axis (str): The axis name (e.g., 'pitch', 'roll', etc.).
            gain (str): The gain name (e.g., 'kp', 'ki', 'kd').
            value (float): The value to set for the specified gain.
        """
        # Construct the attribute path dynamically
        pid_attr = getattr(self.pid, axis)  # e.g., self.pid.pitch
        if not hasattr(pid_attr, gain):
            raise AttributeError(f"The gain '{gain}' does not exist on the PID axis '{axis}'.")
        setattr(pid_attr, gain, value)
    
    def _update_pid_from_param(self, param: DTParam):
        # The parameter name structure is "/ROBOT_NAME/pid_controller_node/pitch_kp"
        axis, gain, = param.name.split("/")[-1].split("_")
        self._set_pid_gain(axis, gain, param.value)
    
    def initialize_pid_gains(self):
        """
        Method that registers the PID parameters DTParams.
        """
        
        # Create the PID parameters
        self.pitch_kp = DTParam("~pitch_kp", param_type=ParamType.FLOAT)
        self.pitch_ki = DTParam("~pitch_ki", param_type=ParamType.FLOAT)
        self.pitch_kd = DTParam("~pitch_kd", param_type=ParamType.FLOAT)

        self.roll_kp = DTParam("~roll_kp", param_type=ParamType.FLOAT)
        self.roll_ki = DTParam("~roll_ki", param_type=ParamType.FLOAT)
        self.roll_kd = DTParam("~roll_kd", param_type=ParamType.FLOAT)

        self.throttle_kp = DTParam("~throttle_kp", param_type=ParamType.FLOAT)
        self.throttle_ki = DTParam("~throttle_ki", param_type=ParamType.FLOAT)
        self.throttle_kd = DTParam("~throttle_kd", param_type=ParamType.FLOAT)

        self.yaw_kp = DTParam("~yaw_kp", param_type=ParamType.FLOAT)
        self.yaw_ki = DTParam("~yaw_ki", param_type=ParamType.FLOAT)
        self.yaw_kd = DTParam("~yaw_kd", param_type=ParamType.FLOAT)
        
        # Add callbacks to update the PID parameters from the DTParams when there is a change
        for param in [self.pitch_kp, self.pitch_ki, self.pitch_kd,
                      self.roll_kp, self.roll_ki, self.roll_kd,
                      self.throttle_kp, self.throttle_ki, self.throttle_kd,
                      self.yaw_kp, self.yaw_ki, self.yaw_kd]:

            self._update_pid_from_param(param)
            param.register_update_callback(lambda p=param: self._update_pid_from_param(p))

    def takeoff_srv(self, req: SetBoolRequest):
        """ Service to switch between flying and not flying """
        if req.data:
            self.current_mode = DroneMode.FLYING
        else:
            self.current_mode = DroneMode.ARMED
        return SetBoolResponse(success=True, message="Mode set to %s" % self.current_mode)
    
    def land_srv(self, req):
        self.set_mode_service(custom_mode="LAND")
        return SetBoolResponse(success=True, message="Mode set to LAND")

    def set_mode(self, mode : str):
        # Call the service
        response = self.set_mode_service(custom_mode=mode)
        
        # Check the response
        if response.mode_sent:
            rospy.loginfo("Successfully called the set mode service")
        else:
            rospy.logwarn("Failed to call the set mode service")

    # ROS SUBSCRIBER CALLBACK METHODS
    #################################
    def current_state_callback(self, state : Odometry):
        """ Store the drone's current state for calculations """
        self.previous_state = self.current_state
        self.current_state = state
        self.state_to_three_dim_vec_structs()

    def desired_pose_callback(self, msg):
        """ Update the desired pose """

        # store the previous desired position
        self.last_desired_position = self.desired_position

        # --- set internal desired pose equal to the desired pose ros message ---

        if self.absolute_desired_position:
            self.desired_position.x = msg.position.x
            self.desired_position.y = msg.position.y
            self.desired_position.z = msg.position.z if 0 <= msg.position.z <= self.max_height * 0.8 else self.last_desired_position.z

        # RELATIVE desired x, y to the CURRENT pose, but
        # RELATIVE desired z to the PREVIOUS DESIRED z (so it appears more responsive)
        else:
            self.desired_position.x = self.current_position.x + msg.position.x
            self.desired_position.y = self.current_position.y + msg.position.y
            # (doesn't limit the mag of the error)
            desired_z = self.last_desired_position.z + msg.position.z
            # the desired z must be above 0 and below the range of the ir sensor (.55meters)
            self.desired_position.z = desired_z if 0 <= desired_z <= self.max_height * 0.8 else self.last_desired_position.z

        if self.desired_position != self.last_desired_position:
            # desired pose changed, the drone should move
            self.moving = True
            rospy.loginfo('moving')
        
        self._desired_height_pub.publish(self.desired_position.z)

    def desired_twist_callback(self, msg):
        """ Update the desired twist """
        self.desired_velocity.x = msg.linear.x
        self.desired_velocity.y = msg.linear.y
        self.desired_velocity.z = msg.linear.z
        self.desired_yaw_velocity = msg.angular.z
        self.desired_velocity_start_time = None
        self.desired_yaw_velocity_start_time = None
        
        rospy.loginfo(f"Desired_velocity set to {self.desired_velocity}")
        
        if self.path_planning:
            self.calculate_travel_time()

    def current_mode_callback(self, msg : FCUState):
        """ Update the current mode """
        if msg.armed:
            if self.current_mode == DroneMode.DISARMED:
                self.current_mode = DroneMode.ARMED
        else:
            self.current_mode = DroneMode.DISARMED
        self.loginfo(f"Current mode set to: {self.current_mode}")

    def position_control_callback(self, msg):
        """ Set whether or not position control is enabled """
        self.position_control = msg.data
        if self.position_control:
            self.desired_position = self.current_position
        if self.position_control != self.last_position_control:
            self.loginfo(f"Position control: {self.position_control}")
            self.last_position_control = self.position_control

    def reset_callback(self, _):
        """ Reset the desired and current poses of the drone and set
        desired velocities to zero """
        self.current_position = Position(z=self.current_position.z)
        self.desired_position = self.current_position
        self.desired_velocity.x = 0
        self.desired_velocity.y = 0

    def lost_callback(self, msg):
        self.lost = msg.data

    def step(self):
        """ Returns the commands generated by the pid """
        time = rospy.get_time()

        self.calc_error()
        if self.position_control:
            if self.position_error.xy_magnitude() < self.safety_threshold and not self.lost:
                if self.moving:
                    if self.position_error.magnitude > 0.05:
                        self.pid_error -= self.velocity_error
                    else:
                        self.moving = False
                        rospy.loginfo('not moving')
            else:
                self.position_control_pub.publish(False)

        if self.desired_velocity.magnitude > 0 or abs(self.desired_yaw_velocity) > 0:
            self.adjust_desired_velocity()

        control_command = self.pid.step(self.pid_error, time, self.desired_yaw_velocity)

        if self.debug_pub.get_num_connections() > 0:
            self._publish_debug_info(control_command)

        return control_command

    # HELPER METHODS
    ################
    def state_to_three_dim_vec_structs(self):
        """
        Convert the values from the state estimator into ThreeDimVec structs to
        make calculations concise
        """
        # store the positions
        pose = self.current_state.pose.pose
        self.current_position.x = pose.position.x
        self.current_position.y = pose.position.y
        self.current_position.z = pose.position.z

        # store the linear velocities
        twist = self.current_state.twist.twist
        self.current_velocity.x = twist.linear.x
        self.current_velocity.y = twist.linear.y
        self.current_velocity.z = twist.linear.z

        # store the orientations
        self.previous_rpy = self.current_rpy
        quaternion = (
            pose.orientation.x,
            pose.orientation.y,
            pose.orientation.z,
            pose.orientation.w
        )
        r, p, y = tf.transformations.euler_from_quaternion(quaternion)
        self.current_rpy = RPY(r, p, y)

    def adjust_desired_velocity(self):
        """ Set the desired velocity back to 0 once the drone has traveled the
        amount of time that causes it to move the specified desired velocity
        travel distance if path_planning otherwise just set the velocities back
        to 0 after the . This is an open loop method meaning that the specified
        travel distance cannot be guarenteed. If path planning_planning is false,
        just set the velocities back to zero, this allows the user to move the
        drone for as long as they are holding down a key
        """
        curr_time = rospy.get_time()
        # set the desired planar velocities to zero if the duration is up
        if self.desired_velocity_start_time is not None:
            # the amount of time the set point velocity is not zero
            duration = curr_time - self.desired_velocity_start_time
            if duration > self.desired_velocity_travel_time:
                self.desired_velocity.x = 0
                self.desired_velocity.y = 0
                self.desired_velocity_start_time = None
        else:
            self.desired_velocity_start_time = curr_time

        # set the desired yaw velocity to zero if the duration is up
        if self.desired_yaw_velocity_start_time is not None:
            # the amount of time the set point velocity is not zero
            duration = curr_time - self.desired_yaw_velocity_start_time
            if duration > self.desired_yaw_velocity_travel_time:
                self.desired_yaw_velocity = 0
                self.desired_yaw_velocity_start_time = None
        else:
            self.desired_yaw_velocity_start_time = curr_time

    # TODO: refactor error computation method
    def calc_error(self):
        """
        Calculate the error in velocity, and if in position hold, add the
        error from lr_pid and fb_pid to the velocity error to control the
        position of the drone
        """
        # store the time difference
        pose_dt = 0
        if self.last_pose_time is not None:
            pose_dt = rospy.get_time() - self.last_pose_time
        self.last_pose_time = rospy.get_time()
        
        self.velocity_error = self.desired_velocity - self.current_velocity
        dz = self.desired_position.z - self.current_position.z
        
        self.pid_error.x = self.velocity_error.x
        self.pid_error.y = self.velocity_error.y
        self.pid_error.z = dz
        
        self.position_error = self.desired_position - self.current_position
        
        if self.position_control:
            # calculate a value to add to the velocity error based based on the
            # position error in the x (roll) direction
            lr_step = self.lr_pid.step(self.position_error.x, pose_dt)
            # calculate a value to add to the velocity error based based on the
            # position error in the y (pitch) direction
            fb_step = self.fb_pid.step(self.position_error.y, pose_dt)
            self.pid_error.x += lr_step
            self.pid_error.y += fb_step

    def calculate_travel_time(self):
        """ return the amount of time that desired velocity should be used to
        calculate the error in order to move the drone the specified travel
        distance for a desired velocity
        """
        if self.desired_velocity.magnitude > 0:
            # time = distance / velocity
            travel_time = self.desired_velocity_travel_distance / self.desired_velocity.xy_magnitude()
        else:
            travel_time = 0.0
        self.desired_velocity_travel_time = travel_time

    def reset(self):
        """ Set desired_position to be current position on `xy` and hover_height on `z`, set
        filtered_desired_velocity to be zero, and reset both the PositionPID
        and VelocityPID.
        """

        self.position_error = Error(0, 0, 0)
        self.desired_position = Position(self.current_position.x, self.current_position.y, self.hover_height)
        
        self.velocity_error = Error(0, 0, 0)
        self.desired_velocity = Velocity(0, 0, 0)

        self.pid.reset()
        self.lr_pid.reset()
        self.fb_pid.reset()

    def publish_control_cmd(self, cmd : List[int]):
        """ Publish the control commands to the drone.

        Args:
            cmd (List[int]): [roll, pitch, yaw, throttle] commands for the drone.
        """
        msg = RC(
            roll=cmd[0],
            pitch=cmd[1],
            yaw=cmd[2],
            throttle=cmd[3],
        )

        self.cmd_pub.publish(msg)
        
    def _publish_debug_info(self, control_command : List[int]):
        """
        Method publishing debug info to  the debug topic.
        """
        debug_info = PIDDebugInfo(
            current_position=self.current_position.as_ros_vector3(),
            desired_position=self.desired_position.as_ros_vector3(),
            position_error=self.position_error.as_ros_vector3(),
            current_velocity=self.current_velocity.as_ros_vector3(),
            desired_velocity=self.desired_velocity.as_ros_vector3(),  
            velocity_error=self.velocity_error.as_ros_vector3(),    
            pid_error=self.pid_error.as_ros_vector3(),
            fly_command=RC(*control_command),
            throttle_integral=self.pid.throttle.integral,
            pid_pitch=self.pid_axis_to_pid_state_msg(self.pid.pitch, self.pid_error.y),
            pid_roll=self.pid_axis_to_pid_state_msg(self.pid.roll, self.pid_error.x),
            pid_throttle=self.pid_axis_to_pid_state_msg(self.pid.throttle, self.pid_error.z),
            pid_yaw=self.pid_axis_to_pid_state_msg(self.pid.yaw, self.desired_yaw_velocity),
        )
        
        self.debug_pub.publish(debug_info)

    @staticmethod
    def pid_axis_to_pid_state_msg(pid : PIDaxis, error : float) -> PIDState:
        """
        Generate a PIDState message from a PIDaxis object for debugging purposes.
        """
        
        msg = PIDState(
            error=error,
            kp=pid.kp,
            ki=pid.ki,
            kd=pid.kd,
            proportional_output=pid._p,
            derivative_output=pid._d,
            integral_output=pid.integral,
        )

    def log_mode_transition(self):
        """
        Method that logs the transition between two flight modes.
        """
        self.loginfo(f"Transitioned from {self.previous_mode} to {self.current_mode}")

    def safety_check(self):
        """
        Method that checks if the drone is within the specified maximum height.
        """
        if self.current_state.pose.pose.position.z > self.max_height:
            self.loginfo("\n disarming because drone is too high \n")
            self.previous_mode = DroneMode.DISARMED
            self.current_mode = DroneMode.DISARMED
            return False
        
        return True

def main(controller : PIDControllerNode):
    verbose = 2

    pid : PIDControllerNode = controller

    control_loop_rate = rospy.Rate(pid.frequency)
    rospy.loginfo('PID Controller Started')

    while not pid.is_shutdown and pid.safety_check():
        pid.heartbeat_pub.publish(Empty())

        # If we are not flying, this can be used to
        # examine the behavior of the PID based on published values
        fly_command = pid.step()

        if pid.previous_mode == DroneMode.DISARMED:
            if pid.current_mode == DroneMode.ARMED:
                pid.reset()
                pid.position_control_pub.publish(False)
                
                pid.log_mode_transition()
                pid.previous_mode = pid.current_mode

        if pid.previous_mode == DroneMode.ARMED:
            if pid.current_mode == DroneMode.FLYING:
                pid.reset()
                
                pid.log_mode_transition()
                pid.previous_mode = pid.current_mode

            elif pid.current_mode == DroneMode.DISARMED:
                pid.log_mode_transition()
                pid.previous_mode = pid.current_mode

        elif pid.previous_mode == DroneMode.FLYING:
            if pid.current_mode == DroneMode.FLYING:
                pid.publish_control_cmd(fly_command)

            elif pid.current_mode == DroneMode.DISARMED:
                pid.log_mode_transition()
                pid.previous_mode = pid.current_mode


                # after flying, take the converged low i terms and set these as the
                # initial values, this allows the drone to "learn" and get steadier
                # with each flight until it converges
                pid.pid.roll_low.init_i = pid.pid.roll_low.integral
                pid.pid.pitch_low.init_i = pid.pid.pitch_low.integral
                # Uncomment below statements to print the converged values.
                # Make sure verbose = 0 so that you can see these values
                if verbose >= 2:
                    pid.logdebug(f'roll_low.init_i {pid.pid.roll_low.init_i}')
                    pid.logdebug(f'pitch_low.init_i {pid.pid.pitch_low.init_i}')
                # ---
                pid.loginfo("Detected state change: FLYING -> DISARMED")
                pid.previous_mode = pid.current_mode

        control_loop_rate.sleep()


if __name__ == '__main__':
    main(PIDControllerNode())
