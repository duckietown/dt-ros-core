#!/usr/bin/env python3
import json

import numpy as np

import rospy
from cv_bridge import CvBridge
from duckietown.dtros import DTROS, NodeType, TopicType
from duckietown_msgs.msg import FSMState, LanePose, SegmentList, Twist2DStamped
from duckietown_msgs.msg import Segment as SegmentMsg
from sensor_msgs.msg import Image, CompressedImage
from std_msgs.msg import String

from dt_state_estimation.lane_filter import (
    LaneFilterHistogram,
    # ILaneFilter,
)
from dt_state_estimation.lane_filter.types import (
    Segment,
    SegmentPoint,
    SegmentColor,
)
from dt_state_estimation.lane_filter.rendering import (
    # plot_belief,
    plot_d_phi,  # new rendering
)
from typing import List


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
        ~car_cmd (:obj:`Twist2DStamped`): The car commands executed. Used for the predict step of the filter
        ~change_params (:obj:`String`): A topic to temporarily changes filter parameters for a finite time
        only
        ~switch (:obj:``BoolStamped): A topic to turn on and off the node. WARNING : to be replaced with a
        service call to the provided mother node switch service
        ~fsm_mode (:obj:`FSMState`): A topic to change the state of the node. WARNING : currently not
        implemented

    Publishers:
        ~lane_pose (:obj:`LanePose`): The computed lane pose estimate
        ~debug/belief_img/compressed (:obj:`CompressedImage`): A debug image that shows the filter's internal state
        ~seglist_filtered (:obj:``SegmentList): a debug topic to send the filtered list of segments that
        are considered as valid

    """

    filter: LaneFilterHistogram
    bridge: CvBridge

    def __init__(self, node_name):
        super(LaneFilterNode, self).__init__(node_name=node_name, node_type=NodeType.PERCEPTION,
                                             fsm_controlled=True)

        self._filter = rospy.get_param("~lane_filter_histogram_configuration", None)
        self._debug = rospy.get_param("~debug", False)

        # Create the filter
        self.filter = LaneFilterHistogram(**self._filter)

        # Creating cvBridge
        self.bridge = CvBridge()

        self.t_last_update = rospy.get_time()
        self.currentVelocity = None

        self.latencyArray = []

        # Subscribers
        self.sub = rospy.Subscriber("~segment_list", SegmentList, self.cbProcessSegments, queue_size=1)

        self.sub_velocity = rospy.Subscriber("~car_cmd", Twist2DStamped, self.updateVelocity)

        self.sub_change_params = rospy.Subscriber("~change_params", String, self.cbTemporaryChangeParams)

        # Publishers
        self.pub_lane_pose = rospy.Publisher(
            "~lane_pose", LanePose, queue_size=1, dt_topic_type=TopicType.PERCEPTION
        )

        self.pub_belief_img = rospy.Publisher(
            "~debug/belief_img/compressed", CompressedImage, queue_size=1, dt_topic_type=TopicType.DEBUG
        )

        self.pub_seglist_filtered = rospy.Publisher(
            "~seglist_filtered", SegmentList, queue_size=1, dt_topic_type=TopicType.DEBUG
        )

        # FSM
        # self.sub_switch = rospy.Subscriber(
        #     "~switch", BoolStamped, self.cbSwitch, queue_size=1)
        self.sub_fsm_mode = rospy.Subscriber("~fsm_mode", FSMState, self.cbMode, queue_size=1)

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

    def cbTemporaryChangeParams(self, msg):
        """Callback that changes temporarily the filter's parameters.

        Args:
            msg (:obj:`String`): list of the new parameters

        """
        # This weird callback changes parameters only temporarily - used in the unicorn intersection.
        # comment from 03/2020
        data = json.loads(msg.data)
        params = data["params"]
        reset_time = data["time"]
        # Set all paramters which need to be updated
        for param_name in list(params.keys()):
            param_val = params[param_name]
            params[param_name] = eval("self.filter." + str(param_name))  # FIXME: really?
            exec("self.filter." + str(param_name) + "=" + str(param_val))  # FIXME: really?

        # Sleep for reset time
        rospy.sleep(reset_time)

        # Reset parameters to old values
        for param_name in list(params.keys()):
            param_val = params[param_name]

            exec("self.filter." + str(param_name) + "=" + str(param_val))  # FIXME: really?

    #    def nbSwitch(self, switch_msg):
    #        """Callback to turn on/off the node
    #
    #        Args:
    #            switch_msg (:obj:`BoolStamped`): message containing the on or off command
    #
    #        """
    #        # All calls to this message should be replaced directly by the srvSwitch
    #        request = SetBool()
    #        request.data = switch_msg.data
    #        eelf.nub_switch(request)

    def cbProcessSegments(self, segment_list_msg):
        """Callback to process the segments

        Args:
            segment_list_msg (:obj:`SegmentList`): message containing list of processed segments

        """
        # Get actual timestamp for latency measurement
        timestamp_before_processing = rospy.Time.now()

        # Step 1: predict
        current_time = rospy.get_time()
        if self.currentVelocity:
            dt = current_time - self.t_last_update
            self.filter.predict(delta_t=dt, v=self.currentVelocity.v, w=self.currentVelocity.omega)

        self.t_last_update = current_time

        segs: List[Segment] = []
        s_msg: SegmentMsg
        for s_msg in segment_list_msg.segments:
            segs.append(self._seg_msg_to_custom_type(s_msg))

        # Step 2: update
        self.filter.update(segs)

        # Step 3: build messages and publish things
        d_max, phi_max = self.filter.get_estimate()
        # self.logdebug(f"estimation: {d_max}, {phi_max}")

        # Getting the highest belief value from the belief matrix
        max_val = self.filter.get_max()
        # Comparing it to a minimum belief threshold to make sure we are certain enough of our estimate
        in_lane = max_val > self.filter.min_max

        # build lane pose message to send
        lane_pose = LanePose()
        lane_pose.header.stamp = segment_list_msg.header.stamp
        lane_pose.d = d_max
        lane_pose.phi = phi_max
        lane_pose.in_lane = in_lane
        # XXX: is it always NORMAL?
        lane_pose.status = lane_pose.NORMAL

        self.pub_lane_pose.publish(lane_pose)

        # # old rendering
        # debug_img_msg = self.bridge.cv2_to_compressed_imgmsg(plot_belief(filter=self.filter))
        if self.pub_belief_img.get_num_connections() > 0:
            debug_img_msg = self.bridge.cv2_to_compressed_imgmsg(plot_d_phi(d=d_max, phi=phi_max))
            debug_img_msg.header = segment_list_msg.header
            self.pub_belief_img.publish(debug_img_msg)

    def cbMode(self, msg):
        return  # TODO adjust self.active

    def updateVelocity(self, twist_msg):
        self.currentVelocity = twist_msg


if __name__ == "__main__":
    lane_filter_node = LaneFilterNode(node_name="lane_filter_node")
    rospy.spin()
