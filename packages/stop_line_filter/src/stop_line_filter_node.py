#!/usr/bin/env python3
import numpy as np

import rospy
from duckietown.dtros import DTParam, DTROS, NodeType, ParamType
from duckietown_msgs.msg import BoolStamped, FSMState, LanePose, SegmentList, StopLineReading
from geometry_msgs.msg import Pose2D

class StopLineFilterNode(DTROS):
    """
    Calculates the relative pose of the robot to the stop line and emits an "at_stop_line" boolean if
    the robot is close enough

    Args:
        node_name (:obj:`str'): a unique, descriptive name for the node that ROS will use

    Configuration:
        stop_distance (:obj:`float'): How far from the stop line should we be in order to output the "at_stop_line"
        output
        min_segs (:obj:`int'): The minimum number of red line segments that should be detected to constitute a red line
        detection
        off_time (:obj:`float'): An amount of time to disable the stop line detection after we leave the stop
        line and start navigating the intersection. Used so that the robot doesn't immediately redetect that it
        is at a stop lane before navigating the intersection
        max_y (:obj:`float'): The maximum offset in the `y` direction (orthogonal to the direction of the lane) to be
        considered a valid stop line. Used so that we don't stop and stop lines in adjacent lanes

    Subscribers:
        ~segment_list (:obj:`SegmentList`): The detected line segments from the line detector
        TODO: subscribe to encoders to update the stop line pose after the stop line is out of the FOV
        ~lane_pose (:obj:`LanePose'): The Lane Pose output from the lane filter
        ~fsm_node/mode (:obj:`FSMState'): Our current state

    Publishers:
        ~stop_line_reading (:obj:`StopLineReading'): Contains booleans for whether a stop line is detected and whether
        whether we are the stop line (according to the stop distance), and a Pose2D that is our best estimate of the
        robot pose relative to the stop line. I.e., the coordinate frame being centered on the middle of the
        stop line with x going forward (down the lane), y going to the left (similar to lane filter), and theta going
        from x to y (Right-hand rule)

    """
    def __init__(self, node_name):
        # Initialize the DTROS parent class
        super(StopLineFilterNode, self).__init__(
            node_name=node_name,
            node_type=NodeType.PERCEPTION,
            fsm_controlled=True)

        # Initialize the parameters
        self.stop_distance = DTParam("~stop_distance", param_type=ParamType.FLOAT)
        self.min_segs = DTParam("~min_segs", param_type=ParamType.INT)
        self.off_time = DTParam("~off_time", param_type=ParamType.FLOAT)
        self.max_y = DTParam("~max_y", param_type=ParamType.FLOAT)

        ## state vars
        self.lane_pose = LanePose()


        ## publishers and subscribers
        self.sub_segs = rospy.Subscriber("~segment_list", SegmentList, self.cb_segments)
        self.sub_lane = rospy.Subscriber("~lane_pose", LanePose, self.cb_lane_pose)
        self.pub_stop_line_reading = rospy.Publisher("~stop_line_reading", StopLineReading, queue_size=1, latch=True)
        self.pub_at_stop_line = rospy.Publisher("~at_stop_line", BoolStamped, queue_size=1)

    def cb_lane_pose(self, lane_pose_msg):
        self.lane_pose = lane_pose_msg

    def cb_segments(self, segment_list_msg):


        good_seg_count = 0
        stop_line_x_accumulator = 0.0
        for segment in segment_list_msg.segments:
            if segment.color != segment.RED:
                continue
            if segment.points[0].x < 0 or segment.points[1].x < 0:  # the point is behind us
                continue

            p1_lane = self.to_lane_frame(segment.points[0])
            p2_lane = self.to_lane_frame(segment.points[1])
            avg_x = 0.5 * (p1_lane[0] + p2_lane[0])
            avg_y = 0.5 * (p1_lane[1] + p2_lane[1])

            # If the line is more than max_y offset in the y direction then it is
            # not a stop line in our lane and we shouldn't count it
            if np.abs(avg_y) > self.max_y.value:
                continue
            stop_line_x_accumulator += avg_x
            good_seg_count += 1.0

        stop_line_reading_msg = StopLineReading()
        stop_line_reading_msg.header.stamp = segment_list_msg.header.stamp
        if good_seg_count < self.min_segs.value:
            stop_line_reading_msg.stop_line_detected = False
            stop_line_reading_msg.at_stop_line = False
            self.pub_stop_line_reading.publish(stop_line_reading_msg)

        else:
            stop_line_reading_msg.stop_line_detected = True
            stop_pose = Pose2D()
            stop_pose.x = - stop_line_x_accumulator / good_seg_count
            stop_pose.y = self.lane_pose.d
            stop_pose.theta = self.lane_pose.phi
            stop_line_reading_msg.stop_pose = stop_pose

            # Only detect redline if y is within max_y distance:
            stop_line_reading_msg.at_stop_line = \
                -stop_pose.x < self.stop_distance.value

            self.pub_stop_line_reading.publish(stop_line_reading_msg)
            if stop_line_reading_msg.at_stop_line:
                msg = BoolStamped()
                msg.header.stamp = stop_line_reading_msg.header.stamp
                msg.data = True
                self.pub_at_stop_line.publish(msg)

    def to_lane_frame(self, point):
        p_homo = np.array([point.x, point.y, 1])
        phi = self.lane_pose.phi
        d = self.lane_pose.d
        T = np.array([[np.cos(phi), -np.sin(phi), 0], [np.sin(phi), np.cos(phi), d], [0, 0, 1]])
        p_new_homo = T.dot(p_homo)
        p_new = p_new_homo[0:2]
        return p_new


if __name__ == "__main__":
    lane_filter_node = StopLineFilterNode(node_name="stop_line_filter")
    rospy.spin()
