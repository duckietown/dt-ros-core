from abc import ABC, abstractmethod
import rospy
from nav_msgs.msg import Odometry
from std_msgs.msg import Header
from geometry_msgs.msg import PoseStamped, TwistStamped
from sensor_msgs.msg import Range, Imu

class StateEstimatorAbs(ABC):
    ''' Abstract base class for state estimators.'''

    def __init__(self):
        ''' Initialize the State '''
        header = Header()
        header.stamp = rospy.Time.now()
        header.frame_id = 'Body'

        self.state = Odometry()
        self.state.header = header

        # Flags for received data
        self.received_twist_data = False
        self.received_range_data = False
        self.received_imu_data = False

    @abstractmethod
    def initialize_estimator(self):
        '''Initialize specific estimator parameters.'''
        pass
    
    @abstractmethod
    def process_pose(self, pose_data : PoseStamped):
        '''Process incoming pose data.'''
        pass

    @abstractmethod
    def process_twist(self, twist_data : TwistStamped):
        '''Process incoming twist data.'''
        pass

    @abstractmethod
    def process_range(self, range_data : Range):
        '''Process incoming range data.'''
        pass
    
    @abstractmethod
    def process_imu(self, imu_data : Imu):
        """
        Process the incoming IMU data.
        """
        pass

    @abstractmethod
    def compute_prior(self):
        '''Predict the state based on the estimator model.'''
        pass
    
    def _ready_to_estimate(self):
        '''Check if enough data has been received to start estimation.'''
        return self.received_range_data and self.received_imu_data

    def update_state(self) -> Odometry:
        """
        Update the state based on the received data.
        """
        return self.state