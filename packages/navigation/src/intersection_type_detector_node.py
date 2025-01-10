#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math

import numpy

import rospy
from duckietown_msgs.msg import AprilTagsWithInfos, FSMState, TurnIDandType, BoolStamped
from duckietown.dtros import DTROS, NodeType, TopicType, DTParam, ParamType


class IntersectionTypeDetectorNode(DTROS):
    def __init__(self, node_name):
        super(IntersectionTypeDetectorNode, self).__init__(
            node_name=node_name,
            node_type=NodeType.PERCEPTION,
            fsm_controlled=True
        )

        # Save the name of the node
        self.node_name = node_name
        self.turn_type = -1
        rospy.loginfo(f"[{self.node_name}] Initializing.")

        # Setup publishers
        self.pub_stop_sign = rospy.Publisher(
            "~stop_sign_intersection_detected",
            BoolStamped,
            queue_size=1,
            latch=True
        )

        self.pub_traffic_light = rospy.Publisher(
            "~traffic_light_intersection_detected",
            BoolStamped,
            queue_size=1,
            latch=True
        )

        # Setup subscribers
        self.sub_topic_tag = rospy.Subscriber("~tag", AprilTagsWithInfos, self.cbTag, queue_size=1)

        rospy.loginfo(f"[{self.node_name}] Initialzed.")


    def cbTag(self, tag_msgs):
            # loop through list of april tags to
            # find the nearest apriltag
            dis_min = 999
            idx_min = -1
            for idx, taginfo in enumerate(tag_msgs.infos):
                if taginfo.tag_type == taginfo.SIGN:
                    if (taginfo.traffic_sign_type == taginfo.STOP or
                        taginfo.traffic_sign_type == taginfo.T_LIGHT_AHEAD):
                        tag_det = (tag_msgs.detections)[idx]
                        pos = tag_det.transform.translation
                        distance = math.sqrt(pos.x**2 + pos.y**2 + pos.z**2)
                        if distance < dis_min:
                            dis_min = distance
                            idx_min = idx

            if idx_min == -1:
                rospy.logwarn("[INTERSECTION_TYPE_DETECTOR_NODE]: Unable to determine intersection type, "
                              "no traffic light or stop sign signs detected")
            else:
                header = tag_msgs.header
                to_pub = BoolStamped()
                to_pub.header=header
                taginfo = (tag_msgs.infos)[idx_min]
                if taginfo.traffic_sign_type == taginfo.STOP:
                    to_pub.data = True
                    self.pub_stop_sign.publish(to_pub)
                elif taginfo.traffic_sign_type == taginfo.T_LIGHT_AHEAD:
                    to_pub.data = True
                    self.pub_traffic_light.publish(to_pub)
                else:
                    rospy.logerr("Something went wrong - tag detection type "
                                 "is not an intersection type")

    def setupParameter(self, param_name, default_value):
        value = rospy.get_param(param_name, default_value)
        rospy.set_param(param_name, value)  # Write to parameter server for transparancy
        # rospy.loginfo("[%s] %s = %s " %(self.node_name,param_name,value))
        return value

    def on_shutdown(self):
        rospy.loginfo(f"[{self.node_name}] Shutting down.")


if __name__ == "__main__":
    # Create the NodeName object
    node = IntersectionTypeDetectorNode(node_name="intersection_type_detector_node")

    # Setup proper shutdown behavior
    rospy.on_shutdown(node.on_shutdown)
    # Keep it spinning to keep the node alive
    rospy.spin()
