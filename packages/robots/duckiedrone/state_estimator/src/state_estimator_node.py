#!/usr/bin/env python3


from duckietown.dtros.dtparam import DTParam, ParamType
from duckietown.dtros.dtros import DTROS, NodeType
import rospy
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu, Range
from geometry_msgs.msg import PoseStamped, TwistStamped

from scripts import StateEstimatorEMA, StateEstimatorUKF12D, StateEstimatorAbs

class StateEstimatorNode(DTROS):
    """
    The node provides state estimates for the drone using various methods:

        - EMA: uses an exponential moving average
        - UKF with a 2D state vector
        - UKF with a 7D state vector
        - UKF with a 12D state vector
        - MoCap: provides ground truth
        - Simulation (drone_simulator.py): provides simulated ground truth
        
    For incomplete state estimators like the 2D UKF, EMA estimates are used to populate 
    the missing state variables such as the x and y positions.

    Args:
        primary (:obj:`str`): The primary state estimation method to be used. Choices are:
            'ema', 'ukf2d', 'ukf7d', 'ukf12d', 'mocap', 'simulator'.
        others (:obj:`list` of :obj:`str`, optional): Additional state estimation methods 
            to run alongside the primary method for purposes such as visualization or debugging. 
            If not specified, defaults to an empty list.
        ir_throttled (:obj:`bool`, optional): Whether to use the throttled IR altitude topic 
            `/pidrone/altitude_throttle`. Default is `False`.
        imu_throttled (:obj:`bool`, optional): Whether to use the throttled IMU topic 
            `/pidrone/imu_throttle`. Default is `False`.
        optical_flow_throttled (:obj:`bool`, optional): Whether to use the throttled optical flow 
            topic `/pidrone/picamera/twist_throttle`. Default is `False`.
        camera_pose_throttled (:obj:`bool`, optional): Whether to use the throttled camera pose 
            topic `/pidrone/picamera/pose_throttle`. Default is `False`.
        sdim (:obj:`int`, optional): The number of spatial dimensions in which to simulate the 
            drone's motion, if running the drone simulator. Choices are `1`, `2`, or `3`. 
            Default is `1`.
        student_ukf (:obj:`bool`, optional): Whether to use the Student's t-distribution-based 
            UKF. Default is `False`.
        ir_var (:obj:`float`, optional): The variance of the IR sensor to be used in 1D simulation. 
            Default is `None`.
        loop_hz (:obj:`float`, optional): The frequency (in Hz) at which to run the predict-update 
            loop of the UKF. Default is `30.0` Hz.

    Configuration:
        ~primary (:obj:`str`): The primary state estimation method.
        ~others (:obj:`list` of :obj:`str`): Additional state estimation methods.
        ~ir_throttled (:obj:`bool`): Whether to use the throttled IR altitude topic.
        ~imu_throttled (:obj:`bool`): Whether to use the throttled IMU topic.
        ~optical_flow_throttled (:obj:`bool`): Whether to use the throttled optical flow topic.
        ~camera_pose_throttled (:obj:`bool`): Whether to use the throttled camera pose topic.
        ~sdim (:obj:`int`): The number of spatial dimensions for the drone simulation.
        ~student_ukf (:obj:`bool`): Whether to use the Student's t-distribution-based UKF.
        ~ir_var (:obj:`float`): The variance of the IR sensor in 1D simulation.
        ~loop_hz (:obj:`float`): The loop frequency for the UKF predict-update cycle.

    Publisher:
        ~pidrone/state (:obj:`Odometry`): The estimated state of the drone, published at the specified loop frequency.

    """
    
    def __init__(self, primary, others=None, ir_throttled=False, imu_throttled=False,
                 optical_flow_throttled=False, camera_pose_throttled=False,
                 sdim=1, student_ukf=False, ir_var=0.0, state_rate=30.0):
        super(StateEstimatorNode, self).__init__(node_name="state_estimator_node", node_type=NodeType.PERCEPTION)
        self.state_msg = Odometry()
        
        # Parameters initialization using DTParam
        self.ir_throttled = DTParam("~ir_throttled", param_type=ParamType.BOOL, default=ir_throttled)
        self.imu_throttled = DTParam("~imu_throttled", param_type=ParamType.BOOL, default=imu_throttled)
        self.optical_flow_throttled = DTParam("~optical_flow_throttled", param_type=ParamType.BOOL, default=optical_flow_throttled)
        self.camera_pose_throttled = DTParam("~camera_pose_throttled", param_type=ParamType.BOOL, default=camera_pose_throttled)

        self.sdim = DTParam("~sdim", param_type=ParamType.INT, default=sdim)
        self.student_ukf = DTParam("~student_ukf", param_type=ParamType.BOOL, default=student_ukf)
        self.ir_var = DTParam("~ir_var", param_type=ParamType.FLOAT, default=ir_var)
        self.state_rate = DTParam("~state_rate", param_type=ParamType.FLOAT, default=state_rate)

        self.rate_enforcer = rospy.Rate(self.state_rate.value)

        self.primary_estimator = DTParam("~primary", param_type=ParamType.STRING, default=primary)
        self.other_estimators = DTParam("~others", param_type=ParamType.LIST, default=others or [])
        self.ema_params = DTParam("~ema_params", param_type=ParamType.DICT, default={})
        self._state_estimator = self.get_state_estimator() # type: ignore

        # Subscribers
        #############
        rospy.Subscriber('~imu', Imu, self.imu_cb)
        rospy.Subscriber('~range', Range, self.range_cb)
        rospy.Subscriber('~twist', Odometry, self.twist_cb)
        
        rospy.Subscriber('pose_topic', PoseStamped, self.pose_cb)
        
        # Publishers
        self.state_pub = rospy.Publisher('~state', Odometry, queue_size=1,
                                         tcp_nodelay=False)

    def pose_cb(self, msg : PoseStamped):
        self._state_estimator.process_pose(msg)

    def range_cb(self,msg : Range):
        self._state_estimator.process_range(msg)
        
    def imu_cb(self, msg : Imu):
        self._state_estimator.process_imu(msg)

    def twist_cb(self, msg : TwistStamped):
        self._state_estimator.process_twist(msg)

    def get_state_estimator(self) -> StateEstimatorAbs:
        filters_name_to_class = {
            'ema': StateEstimatorEMA,
            'ukf2d': NotImplementedError,
            'ukf7d': NotImplementedError,
            'ukf12d': StateEstimatorUKF12D,
            'mocap': NotImplementedError,
            'simulator': NotImplementedError
        }
        
        # Get the class based on the provided name
        try:
            state_estimator_class = filters_name_to_class[self.primary_estimator.value]
        except KeyError:
            raise ValueError(f"Invalid primary state estimator: {self.primary_estimator.value}")
        
        # Handle NotImplementedError cases
        if state_estimator_class is NotImplementedError:
            raise NotImplementedError(f"{self.primary_estimator.value} is not yet implemented.")
        
        # Ensure the class is a subclass of StateEstimatorAbs
        if not issubclass(state_estimator_class, StateEstimatorAbs):
            raise ValueError(f"Invalid state estimator class: {self.primary_estimator.value}")
        
        self.loginfo(f"Using {self.primary_estimator.value} as the primary state estimator.")

        if state_estimator_class == StateEstimatorEMA:
            return StateEstimatorEMA(
                alpha_pose=self.ema_params.value["alpha_pose"],
                alpha_twist=self.ema_params.value["alpha_twist"],
                alpha_range=self.ema_params.value["alpha_range"],
            )

        return state_estimator_class()
 
    def state_callback(self):
        """
        Callback that handles the primary estimator republishing.
        """
        # TODO: Consider creating a new Odometry message rather than modifying just
        #       one Odometry message
        
        state_msg = self._state_estimator.update_state()
        state_msg.header.stamp = rospy.Time.now()

        self.state_pub.publish(state_msg)
        

    ###############################################################
    # TODO: these should be a debug topic in the base StateEstimatorAbs class or UKF StateEstimator class    
    def ukf_analytics_callback(self, msg):
        self.last_ukf_height = msg.pose.pose.position.z
        if self.last_ground_truth_height is not None:
            stats_msg = UkfStats()
            stats_msg.header.stamp = rospy.Time.now()
            stats_msg.error = self.last_ukf_height - self.last_ground_truth_height
            stats_msg.stddev = (msg.pose.covariance[14])**0.5
            self.ukf_stats_pub.publish(stats_msg)

    def setup_ukf_with_ground_truth(self):
        """
        Determine if a UKF is being run simultaneously with ground truth. This
        informs whether or not we can provide certain analytics about the UKF's
        performance to the user in the web interface
        """
        raise DeprecationWarning
        
        do_setup = (('ukf2d' in self.estimators or 'ukf7d' in self.estimators or
                     'ukf12d' in self.estimators) and
                    ('simulator' in self.estimators or 'mocap' in self.estimators))
        if do_setup:
            
            self.last_ground_truth_height = None
            self.last_ukf_height = None
            self.ukf_stats_pub = rospy.Publisher('ukf_stats', UkfStats, queue_size=1,
                                             tcp_nodelay=False)
            if 'ukf' in self.primary_estimator:
                # If the primary estimator is a UKF, use this one
                ukf_to_use = self.primary_estimator
            else:
                # Search through the other estimators
                possible_ukfs = []
                for estimator in self.other_estimators:
                    if 'ukf' in estimator:
                        possible_ukfs.append(estimator)
                if len(possible_ukfs) > 1:
                    # Give user the option
                    got_good_input = False
                    while not got_good_input:
                        print ('Please enter the list number of which UKF to use'
                               ' for comparison against ground truth:')
                        for num, ukf in enumerate(possible_ukfs):
                            print('{}: {}'.format(num+1, ukf))
                        selection = input()
                        try:
                            selection = int(selection)
                            if selection <= 0 or selection > len(possible_ukfs):
                                raise ValueError
                            ukf_to_use = possible_ukfs[selection]
                            got_good_input = True
                        except ValueError:
                            print('Invalid input.')
                elif len(possible_ukfs) == 1:
                    # This is the only other option; otherwise, do_setup would
                    # be False
                    ukf_to_use = possible_ukfs[0]
            if ukf_to_use == 'ukf2d':
                rospy.Subscriber(self.ukf_topics[2], Odometry, self.ukf_analytics_callback)
            elif ukf_to_use == 'ukf7d':
                rospy.Subscriber(self.ukf_topics[7], Odometry, self.ukf_analytics_callback)
            elif ukf_to_use == 'ukf12d':
                rospy.Subscriber(self.ukf_topics[12], Odometry, self.ukf_analytics_callback)
            for estimator in self.estimators:
                if estimator == 'simulator':
                    topic = self.simulator_topic
                elif estimator == 'mocap':
                    topic = self.mocap_topic
            rospy.Subscriber(topic, StateGroundTruth, self.ground_truth_analytics_callback)

               
    def ground_truth_analytics_callback(self, msg):
        self.last_ground_truth_height = msg.pose.position.z
        # Ground truth value should come in before UKF value. Could perhaps
        # match timestamps or header sequence numbers to check or synchronize.

    ###############################################################

def main():
    state_estimator = StateEstimatorNode(primary='ema')
    
    while not state_estimator.is_shutdown:
        state_estimator.state_callback()
        state_estimator.rate_enforcer.sleep()


if __name__ == "__main__":
    main()


