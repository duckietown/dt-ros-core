#!/usr/bin/env python3

import os
import rospy
from duckietown.dtros import DTROS, NodeType
from duckietown_msgs.msg import AprilTagsWithInfos
from std_msgs.msg import Int32

class ApriltagDetectorSubscriberNode(DTROS):
    def __init__(self, node_name):
        # initialize the DTROS parent class
        super(ApriltagDetectorSubscriberNode, self).__init__(node_name=node_name, node_type=NodeType.GENERIC)
        # construct subscriber
        self.veh = rospy.get_namespace().strip("/")
        self.sub = rospy.Subscriber("apriltag_postprocessing_node/apriltags_out", AprilTagsWithInfos, self.callback, buff_size=10000000, queue_size=1)
        #self.sub = rospy.Subscriber("apriltag_postprocessing_node/tag_id", Int32, self.callback, buff_size=10000000, queue_size=1)

    def callback(self, data):
        rospy.loginfo(f"Subscriber testing DETECT APRILTAGS, {data}")
        #rospy.loginfo(f"Subscriber testing TAG_ID, {data}")
if __name__ == '__main__':
    # create the node
    node = ApriltagDetectorSubscriberNode(node_name='apriltag_detector_subscriber_node')

    # keep spinning
    rospy.spin()