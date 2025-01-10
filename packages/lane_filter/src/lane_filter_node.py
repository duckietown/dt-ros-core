#!/usr/bin/env python3

# USES THE NEW WHEEL ENCODERS DATA

import numpy as np
import rospy
from cv_bridge import CvBridge

from dt_state_estimation.lane_filter.types import (
    Segment,
    SegmentPoint,
    SegmentColor,
)
from duckietown.dtros import DTROS, NodeType, TopicType
from duckietown_msgs.msg import LanePose, SegmentList, WheelEncoderStamped, EpisodeStart
from duckietown_msgs.msg import Segment as SegmentMsg
from sensor_msgs.msg import Image

from lane_filter import LaneFilterHistogram


class LaneFilterNode(DTROS):
    """Generates an estimate of the lane pose.

    Creates a `lane_filter` to get estimates on `d` and `phi`, the lateral and heading deviation from the
    center of the lane.
    It gets the segments extracted by the line_detector as input and output the lane pose estimate.


    Args:
        node_name (:obj:`str`): a unique, descriptive name for the node that ROS will use

    Configuration:
        ~filter (:obj:`list`): A list of parameters for the lane pose estimation filter
        ~debug (:obj:`bool`): A parameter to enable/disable the publishing of debug topics and images

    Subscribers:
        ~segment_list (:obj:`SegmentList`): The detected line segments from the line detector
        ~(left/right)_wheel_encoder_node/tick (:obj: `WheelEncoderStamped`): Information from the wheel encoders\
        ~episode_start (:obj: `EpisodeStart`): The signal that a new episode has started - used to reset the filter

    Publishers:
        ~lane_pose (:obj:`LanePose`): The computed lane pose estimate
        ~debug/belief_img/compressed (:obj:`CompressedImage`): A debug image that shows the filter's internal state
        ~seglist_filtered (:obj:``SegmentList): a debug topic to send the filtered list of segments that
        are considered as valid

    """

    filter: LaneFilterHistogram
    bridge: CvBridge

    def __init__(self, node_name):
        super(LaneFilterNode, self).__init__(
            node_name=node_name,
            node_type=NodeType.PERCEPTION,
            fsm_controlled=True
        )

        self._filter = rospy.get_param("~lane_filter_histogram_configuration", None)
        self._debug = rospy.get_param("~debug", False)
        self._predict_freq = rospy.get_param("~predict_frequency", 30.0)
        #Enocder Init
        self.right_encoder_ticks = 0
        self.right_encoder_initialized = False
        self.left_encoder_ticks = 0
        self.left_encoder_initialized = False
        self.right_encoder_ticks_delta = 0
        self.left_encoder_ticks_delta = 0


        # Load the needed filter parameters defined elsewhere need here
        try:
            encoder_resolution = rospy.get_param("left_wheel_encoder_driver_node/resolution", 135)
            wheel_baseline = rospy.get_param("kinematics_node/baseline")
            wheel_radius = rospy.get_param("kinematics_node/radius")
        except rospy.KeyError as e:
            rospy.logerror(f"[Lane filter] Unable to load required param: {e}")

        # Create the filter
        self.filter = LaneFilterHistogram(encoder_resolution,
                                          wheel_baseline,
                                          wheel_radius,
                                          **self._filter)


        # this is only used for the timestamp of the first publication
        self.last_update_stamp = rospy.Time.now()


        # Creating cvBridge
        self.bridge = CvBridge()


        # Subscribers

        self.sub_segment_list = rospy.Subscriber(
            "~segment_list", SegmentList, self.cbProcessSegments, queue_size=1
        )

        self.sub_encoder_left = rospy.Subscriber(
            "~left_wheel_encoder_driver_node/tick", WheelEncoderStamped, self.cbProcessLeftEncoder, queue_size=1
        )

        self.sub_encoder_right = rospy.Subscriber(
            "~right_wheel_encoder_driver_node/tick", WheelEncoderStamped, self.cbProcessRightEncoder, queue_size=1
        )


        # Publishers
        self.pub_lane_pose = rospy.Publisher(
            "~lane_pose", LanePose, queue_size=1, dt_topic_type=TopicType.PERCEPTION
        )

        # NEW DEBUG RENDERING: to be enabled, also replaces the one below
        # self.pub_belief_img = rospy.Publisher(
        #     "~debug/belief_img/compressed", CompressedImage, queue_size=1, dt_topic_type=TopicType.DEBUG
        # )

        self.pub_belief_img = rospy.Publisher(
            # TODO: this should be a CompressedImage instead
            "~belief_img", Image, queue_size=1, dt_topic_type=TopicType.DEBUG
        )


        # Set up a timer for prediction (if we got encoder data) since that data can come very quickly
  #      rospy.Timer(rospy.Duration(1 / self._predict_freq), self.cbPredict)
        self.publishEstimate(self.last_update_stamp)


    def cbEpisodeStart(self, msg):
        rospy.loginfo("Lane Filter Resetting")
        self.filter.initialize_belief()

    @staticmethod
    def _seg_msg_to_custom_type(msg: SegmentMsg):
        color: SegmentColor = SegmentColor.WHITE
        if msg.color == SegmentMsg.YELLOW:
            color = SegmentColor.YELLOW
        elif msg.color == SegmentMsg.RED:
            color = SegmentColor.RED

        p1, p2 = msg.points

        return Segment(
            color=color,
            points=[
                SegmentPoint(x=p1.x, y=p1.y),
                SegmentPoint(x=p2.x, y=p2.y),
            ],
        )

    def cbProcessLeftEncoder(self, left_encoder_msg):
        # we need to account for the possibility that the encoder is not reading
        # 0 at startup
        if not self.left_encoder_initialized:
            self.left_encoder_ticks = left_encoder_msg.data
            self.left_encoder_initialized = True
        self.left_encoder_ticks_delta = left_encoder_msg.data - self.left_encoder_ticks

    def cbProcessRightEncoder(self, right_encoder_msg):
        if not self.right_encoder_initialized:
            self.right_encoder_ticks = right_encoder_msg.data
            self.right_encoder_initialized = True
        self.right_encoder_ticks_delta = right_encoder_msg.data - self.right_encoder_ticks

    def cbPredict(self):
        if self.left_encoder_ticks_delta == 0 or self.right_encoder_ticks_delta == 0:
            self.publishEstimate(self.last_update_stamp)
            return
        self.filter.predict(self.left_encoder_ticks_delta, self.right_encoder_ticks_delta)
        self.left_encoder_ticks += self.left_encoder_ticks_delta
        self.right_encoder_ticks += self.right_encoder_ticks_delta
        self.left_encoder_ticks_delta = 0
        self.right_encoder_ticks_delta = 0

        self.publishEstimate(self.last_update_stamp)

    def cbProcessSegments(self, segment_list_msg):
        """Callback to process the segments

        Args:
            segment_list_msg (:obj:`SegmentList`): message containing list of processed segments

        """
        self.cbPredict()
        self.last_update_stamp = segment_list_msg.header.stamp

        self.filter.update(segment_list_msg.segments)

        self.publishEstimate(segment_list_msg.header.stamp)

    def publishEstimate(self, timestamp):

        [d_max, phi_max] = self.filter.getEstimate()

        # Getting the highest belief value from the belief matrix
        max_val = self.filter.getMax()
        # Comparing it to a minimum belief threshold to make sure we are certain enough of our estimate
        in_lane = max_val > self.filter.min_max

        # build lane pose message to send
        lanePose = LanePose()
        lanePose.header.stamp = timestamp
        lanePose.d = d_max
        lanePose.phi = phi_max
        lanePose.in_lane = in_lane
        # XXX: is it always NORMAL?
        lanePose.status = lanePose.NORMAL

        self.pub_lane_pose.publish(lanePose)
        if self._debug:
            self.debugOutput()
    def debugOutput(self):
        """Creates and publishes debug messages

        """
        if self._debug:
            # Create belief image and publish it
            belief_img = self.bridge.cv2_to_imgmsg(
                np.array(255 * self.filter.belief).astype("uint8"), "mono8"
            )
            #belief_img.header.stamp = segment_list_msg.header.stamp # FIXME: REPLACE WITH ENCODER TIMESTAMPS MAYBE
            self.pub_belief_img.publish(belief_img)


            # NEW RENDERING: to be enabled
            # if self.pub_belief_img.get_num_connections() > 0:
            #     debug_img_msg = self.bridge.cv2_to_compressed_imgmsg(plot_d_phi(d=d_max, phi=phi_max))
            #     debug_img_msg.header = segment_list_msg.header
            #     self.pub_belief_img.publish(debug_img_msg)

            #FIXME: USE THE Visualization of the lane filter
            #self.filter.get_plot_phi_d()

    def loginfo(self, s):
        rospy.loginfo("[%s] %s" % (self.node_name, s))

if __name__ == "__main__":
    lane_filter_node = LaneFilterNode(node_name="lane_filter_node")
    rospy.spin()
