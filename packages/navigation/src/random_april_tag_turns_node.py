#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math

import numpy

import rospy
from duckietown_msgs.msg import AprilTagsWithInfos, FSMState, TurnIDandType, BoolStamped
from std_msgs.msg import Int16  # Imports msg
from duckietown.dtros import DTROS, NodeType, TopicType, DTParam, ParamType


class RandomAprilTagTurnsNode(DTROS):
    def __init__(self, node_name):
        super(RandomAprilTagTurnsNode, self).__init__(
            node_name=node_name,
            node_type=NodeType.PERCEPTION,
            fsm_controlled=True)

        # Save the name of the node
        self.node_name = node_name
        self.turn_type = -1
        rospy.loginfo(f"[{self.node_name}] Initializing.")

        # Setup publishers
        self.pub_turn_type = rospy.Publisher("~turn_type", Int16, queue_size=1, latch=True)
        self.pub_id_and_type = rospy.Publisher("~turn_id_and_type", TurnIDandType, queue_size=1, latch=True)
        self.pub_intersection_go = rospy.Publisher("~intersection_go", BoolStamped, queue_size=1)

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
                    # we need to make sure it's a sign that tells us topology
                    if taginfo.traffic_sign_type in {
                        taginfo.NO_RIGHT_TURN,
                        taginfo.LEFT_T_INTERSECT,
                        taginfo.NO_LEFT_TURN,
                        taginfo.RIGHT_T_INTERSECT,
                        taginfo.T_INTERSECTION,
                        taginfo.FOUR_WAY
                    }:
                        tag_det = (tag_msgs.detections)[idx]
                        pos = tag_det.transform.translation
                        distance = math.sqrt(pos.x**2 + pos.y**2 + pos.z**2)
                        if distance < dis_min:
                            dis_min = distance
                            idx_min = idx

            if idx_min == -1:
                rospy.logwarn("[RANDOM_APRIL_TAG_TURNS_NODE]: Unable to determine available turns at intersection"
                              "no appropriate signs detected")
            else:
                taginfo = (tag_msgs.infos)[idx_min]

                availableTurns = []
                # go through possible intersection types
                signType = taginfo.traffic_sign_type
                if signType == taginfo.NO_RIGHT_TURN or signType == taginfo.LEFT_T_INTERSECT:
                    availableTurns = [
                        0,
                        1,
                    ]  # these mystical numbers correspond to the array ordering in open_loop_intersection_control_node (very bad)
                elif signType == taginfo.NO_LEFT_TURN or signType == taginfo.RIGHT_T_INTERSECT:
                    availableTurns = [1, 2]
                elif signType == taginfo.FOUR_WAY:
                    availableTurns = [0, 1, 2]
                elif signType == taginfo.T_INTERSECTION:
                    availableTurns = [0, 2]
                # rospy.loginfo(f"[{self.node_name}] reports Available turns are: [{availableTurns}]")
                # now randomly choose a possible direction
                if len(availableTurns) > 0:
                    randomIndex = numpy.random.randint(len(availableTurns))
                    chosenTurn = availableTurns[randomIndex]
                    self.turn_type = chosenTurn
                    self.pub_turn_type.publish(self.turn_type)

                    id_and_type_msg = TurnIDandType()
                    id_and_type_msg.tag_id = taginfo.id
                    id_and_type_msg.turn_type = self.turn_type
                    self.pub_id_and_type.publish(id_and_type_msg)

                    intersection_go_msg = BoolStamped()
                    intersection_go_msg.header = tag_msgs.header
                    intersection_go_msg.data = True
                    self.pub_intersection_go.publish(intersection_go_msg)

    def setupParameter(self, param_name, default_value):
        value = rospy.get_param(param_name, default_value)
        rospy.set_param(param_name, value)  # Write to parameter server for transparancy
        # rospy.loginfo("[%s] %s = %s " %(self.node_name,param_name,value))
        return value

    def on_shutdown(self):
        rospy.loginfo(f"[{self.node_name}] Shutting down.")


if __name__ == "__main__":
    # Initialize the node with rospy
    # rospy.init_node("random_april_tag_turns_node", anonymous=False)

    # Create the NodeName object
    node = RandomAprilTagTurnsNode(node_name="random_april_tag_turns_node")

    # Setup proper shutdown behavior
    rospy.on_shutdown(node.on_shutdown)
    # Keep it spinning to keep the node alive
    rospy.spin()
