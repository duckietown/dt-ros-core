import os
import cv2
import numpy as np
import rospy
import tf
import tf.transformations
from cv_bridge import CvBridge
from duckietown.dtros.constants import NodeType
from duckietown.dtros.dtros import DTROS
from geometry_msgs.msg import PoseStamped
from localization_helper import PROB_THRESHOLD, LocalizationParticleFilter
from nav_msgs.msg import Odometry
from sensor_msgs.msg import CompressedImage
from std_srvs.srv import Trigger, TriggerRequest, TriggerResponse

# ---------- map parameters ----------- #
MAP_PIXEL_WIDTH = 3227  # in pixel
MAP_PIXEL_HEIGHT = 2447
MAP_REAL_WIDTH = 1.4  # in meter
MAP_REAL_HEIGHT = 1.07
# ------------------------------------- #

# ---------- camera parameters ----------- #
# (TODO: These should be retrieved from the ~camera_info topic )
CAMERA_WIDTH = 480
CAMERA_HEIGHT = 640
METER_TO_PIXEL = (
    float(MAP_PIXEL_WIDTH) / MAP_REAL_WIDTH + float(MAP_PIXEL_HEIGHT) / MAP_REAL_HEIGHT
) / 2.0
CAMERA_CENTER = np.array(
    [(CAMERA_WIDTH - 1) / 2.0, (CAMERA_HEIGHT - 1) / 2.0],
    dtype=np.float32,
).reshape(-1, 1, 2)
# ---------------------------------------- #

# ---------- localization parameters ----------- #
MAX_BAD_COUNT = -10
NUM_PARTICLE = 30
NUM_FEATURES = 200
# ---------------------------------------------- #


class Localization(DTROS):
    """
    implements Monte-Carlo Localization using the pi-camera
    """

    def __init__(self, node_name):
        super(Localization, self).__init__(
            node_name=node_name, node_type=NodeType.LOCALIZATION
        )
        self.bridge = CvBridge()
        self.br = tf.TransformBroadcaster()

        self.detector = cv2.ORB.create(
            nfeatures=NUM_FEATURES, scoreType=cv2.ORB_FAST_SCORE
        )
        self.estimator = LocalizationParticleFilter(
            camera_width=CAMERA_WIDTH,
            camera_height=CAMERA_HEIGHT,
            camera_center=CAMERA_CENTER,
            map_image_path=os.path.join(os.path.dirname(__file__), '../../../../../../assets/localization/map.jpg')
        )

        # [x, y, z, yaw]
        self.pos = [0, 0, 0, 0]
        self.posemsg = PoseStamped()

        self.angle_x = 0.0
        self.angle_y = 0.0
        # localization does not estimate z
        self.z = 0.0

        self.first_locate = True
        self.locate_position = False

        self.prev_img = None
        self.prev_kp = None
        self.prev_des = None
        self.prev_time = None
        self.prev_rostime = None

        self.map_counter = 0
        self.max_map_counter = 0

        # constants
        self.alpha_yaw = 0.1  # perceived yaw smoothing alpha
        self.hybrid_alpha = 0.3  # blend position with first frame and int

        self.reset_service = rospy.Service(
            "~reset_transform", Trigger, self.reset_callback
        )
        rospy.Subscriber("~state", Odometry, self.state_callback)
        rospy.Subscriber("~image", CompressedImage, self.image_callback)
        self.posepub = rospy.Publisher("~pose", PoseStamped, queue_size=1)

    def image_callback(self, msg: CompressedImage):
        try:
            # Use compressed_imgmsg_to_cv2 for CompressedImage messages
            curr_img = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding="passthrough")
            # Process the image as needed
        except Exception as e:
            rospy.logerr("Failed to convert image: %s", str(e))

        curr_rostime = rospy.Time.now()
        self.posemsg.header.stamp = curr_rostime
        curr_time = curr_rostime.to_sec()

        # start MCL localization
        if self.locate_position:
            curr_kp, curr_des = self.detector.detectAndCompute(curr_img, None)

            if curr_kp is not None and curr_kp is not None:
                # generate particles for the first time
                if self.first_locate:
                    particle = self.estimator.initialize_particles(
                        NUM_PARTICLE, curr_kp, curr_des
                    )
                    self.first_locate = False
                    self.pos = [
                        particle.x(),
                        particle.y(),
                        particle.z(),
                        particle.yaw(),
                    ]

                    self.posemsg.pose.position.x = particle.x()
                    self.posemsg.pose.position.y = particle.y()
                    self.posemsg.pose.position.z = particle.z()
                    x, y, z, w = tf.transformations.quaternion_from_euler(
                        0, 0, self.pos[3]
                    )

                    self.posemsg.pose.orientation.x = x
                    self.posemsg.pose.orientation.y = y
                    self.posemsg.pose.orientation.z = z
                    self.posemsg.pose.orientation.w = w

                    rospy.loginfo("first %s", particle)
                else:
                    particle = self.estimator.update(
                        self.z,
                        self.angle_x,
                        self.angle_y,
                        self.prev_kp,
                        self.prev_des,
                        curr_kp,
                        curr_des,
                    )

                    # update position
                    self.pos = [
                        self.hybrid_alpha * particle.x()
                        + (1.0 - self.hybrid_alpha) * self.pos[0],
                        self.hybrid_alpha * particle.y()
                        + (1.0 - self.hybrid_alpha) * self.pos[1],
                        self.z,
                        self.alpha_yaw * particle.yaw()
                        + (1.0 - self.alpha_yaw) * self.pos[3],
                    ]

                    self.posemsg.pose.position.x = self.pos[0]
                    self.posemsg.pose.position.y = self.pos[1]
                    self.posemsg.pose.position.z = self.pos[2]
                    x, y, z, w = tf.transformations.quaternion_from_euler(
                        0, 0, self.pos[3]
                    )

                    self.posemsg.pose.orientation.x = x
                    self.posemsg.pose.orientation.y = y
                    self.posemsg.pose.orientation.z = z
                    self.posemsg.pose.orientation.w = w
                    self.posepub.publish(self.posemsg)

                    rospy.loginfo(
                        "--pose %f %f %f", self.pos[0], self.pos[1], self.pos[3]
                    )

                    # if all particles are not good estimations
                    if is_almost_equal(particle.weight(), PROB_THRESHOLD):
                        self.map_counter = self.map_counter - 1
                    elif self.map_counter <= 0:
                        self.map_counter = 1
                    else:
                        self.map_counter = min(self.map_counter + 1, -MAX_BAD_COUNT)

                    # if it's been a while without a significant average weight
                    if self.map_counter < MAX_BAD_COUNT:
                        self.first_locate = True
                        self.map_counter = 0
                        rospy.loginfo("Restart localization")

                    rospy.loginfo("count %d", self.map_counter)
            else:
                self.logwarn("CANNOT FIND ANY FEATURES !!!!!")

            self.prev_kp = curr_kp
            self.prev_des = curr_des

        self.prev_img = curr_img
        self.prev_time = curr_time
        self.prev_rostime = curr_rostime
        self.br.sendTransform(
            (self.pos[0], self.pos[1], self.z),
            tf.transformations.quaternion_from_euler(0, 0, self.pos[3]),
            rospy.Time.now(),
            "base",
            "world",
        )

    def state_callback(self, msg: Odometry):
        """update z, angle x, and angle y data when ~state is published to"""
        self.z = msg.pose.pose.position.z
        # BUG: This is not the correct way to get the orientation
        self.angle_x = msg.twist.twist.angular.x
        self.angle_y = msg.twist.twist.angular.y

    def reset_callback(self, req: TriggerRequest):
        try:
            """Reset localization when '~reset_transform' is published to"""
            self.loginfo("Starting localization")
            self.locate_position = True
            self.first_locate = True
            self.map_counter = 0
            self.max_map_counter = 0
        except Exception as e:
            self.logerr("Failed to reset localization: %s", str(e))
            return TriggerResponse(success=False)
        
        return TriggerResponse(success=True)


def is_almost_equal(x, y):
    epsilon = 1 * 10 ** (-8)
    return abs(x - y) <= epsilon


if __name__ == "__main__":
    localization_node = Localization(node_name="localization_node")
    rospy.spin()
