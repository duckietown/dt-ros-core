import cv2
import numpy as np
import rospy
import tf
from cv_bridge import CvBridge
from duckietown.dtros.constants import NodeType
from duckietown.dtros.dtros import DTROS
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import CompressedImage
from slam_helper import FastSLAM
from std_srvs.srv import Trigger, TriggerResponse, TriggerRequest
from nav_msgs.msg import Odometry
import tf.transformations


# ---------- camera parameters ----------- #
CAMERA_WIDTH = 480
CAMERA_HEIGHT = 640
CAMERA_CENTER = np.float32([(CAMERA_WIDTH - 1) / 2., (CAMERA_HEIGHT - 1) / 2.]).reshape(-1, 1, 2)
# ---------------------------------------- #

# ---------- SLAM parameters ----------- #
NUM_PARTICLE = 20
NUM_FEATURES = 50
# --------------------------------------- #

class SLAMNode(DTROS):
    """
    Node that runs fastSLAM for off-board SLAM using pi camera images.
    """

    def __init__(self, node_name):
        super(SLAMNode, self).__init__(node_name=node_name, node_type=NodeType.LOCALIZATION)
        
        self.bridge = CvBridge()
        self.br = tf.TransformBroadcaster()

        # Publishers and Subscribers
        rospy.Subscriber("~image", CompressedImage, self.image_callback)
        rospy.Subscriber("~state", Odometry, self.state_callback)
        self.posepub = rospy.Publisher("~pose", PoseStamped, queue_size=1)
        
        self.reset_service = rospy.Service("~reset_transform", Trigger, self.reset_callback)

        # ORB Feature Detector
        self.detector = cv2.ORB.create(nfeatures=NUM_FEATURES, scoreType=cv2.ORB_FAST_SCORE)

        # SLAM Estimator
        self.estimator = FastSLAM()

        # State variables
        self.pos = [0, 0, 0, 0]  # x, y, z, yaw
        self.posemsg = PoseStamped()
        self.angle_x = 0.0
        self.angle_y = 0.0
        self.z = 0.0

        self.first_locate = True
        self.locate_position = False

        self.prev_img = None
        self.prev_kp = None
        self.prev_des = None

        # Constants
        self.alpha_yaw = 0.1  # smoothing alpha for yaw
        self.hybrid_alpha = 0.3  # blending position

    def image_callback(self, msg: CompressedImage):
        """Callback to process images and perform SLAM."""
        try:
            curr_img = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding="passthrough")
        except Exception as e:
            rospy.logerr("Failed to convert image: %s", str(e))
            return

        curr_rostime = rospy.Time.now()
        self.posemsg.header.stamp = curr_rostime

        # Start SLAM
        if self.locate_position:
            curr_kp, curr_des = self.detector.detectAndCompute(curr_img, None)

            if curr_kp is not None and len(curr_kp) > 0:
                if self.first_locate:
                    pose = self.estimator.generate_particles(NUM_PARTICLE)
                    self.first_locate = False
                    self.pos = pose
                    self.update_pose(pose)
                    rospy.loginfo("Initial Pose: %s", pose)
                else:
                    pose, weight = self.estimator.run(self.z, self.prev_kp, self.prev_des, curr_kp, curr_des)
                    self.update_position(pose)
                    self.update_pose(self.pos)
                    rospy.loginfo("--Pose: x=%f, y=%f, yaw=%f", self.pos[0], self.pos[1], self.pos[3])
                    rospy.loginfo("--Weight: %f", weight)
            else:
                rospy.logwarn("Cannot find any features")

            self.prev_kp = curr_kp
            self.prev_des = curr_des

        self.prev_img = curr_img
        self.br.sendTransform((self.pos[0], self.pos[1], self.z),
                              tf.transformations.quaternion_from_euler(0, 0, self.pos[3]),
                              rospy.Time.now(), "base", "world")

    def state_callback(self, msg: Odometry):
        """update z, angle x, and angle y data when ~state is published to"""
        self.z = msg.pose.pose.position.z
        # BUG: This is not the correct way to get the orientation
        self.angle_x = msg.twist.twist.angular.x
        self.angle_y = msg.twist.twist.angular.y

    def reset_callback(self, req: TriggerRequest):
        """Resets the SLAM system when reset service is called."""
        rospy.loginfo("Starting localization")
        self.locate_position = True
        self.first_locate = True
        return TriggerResponse(success=True)

    def update_position(self, new_pose):
        """Updates the current estimated position with blending."""
        self.pos = [
            self.hybrid_alpha * new_pose[0] + (1.0 - self.hybrid_alpha) * self.pos[0],
            self.hybrid_alpha * new_pose[1] + (1.0 - self.hybrid_alpha) * self.pos[1],
            self.z,
            self.alpha_yaw * new_pose[3] + (1.0 - self.alpha_yaw) * self.pos[3]
        ]

    def update_pose(self, pose):
        """Publishes the updated pose to the topic."""
        self.posemsg.pose.position.x = pose[0]
        self.posemsg.pose.position.y = pose[1]
        self.posemsg.pose.position.z = self.z
        x, y, z, w = tf.transformations.quaternion_from_euler(0, 0, pose[3])
        self.posemsg.pose.orientation.x = x
        self.posemsg.pose.orientation.y = y
        self.posemsg.pose.orientation.z = z
        self.posemsg.pose.orientation.w = w
        self.posepub.publish(self.posemsg)


if __name__ == "__main__":
    slam_node = SLAMNode(node_name="slam_node")
    rospy.spin()
