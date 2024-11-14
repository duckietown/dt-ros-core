#!/usr/bin/env python3
import numpy as np

import rospy
from cv_bridge import CvBridge
from duckietown.dtros import DTParam, DTROS, NodeType, ParamType
from duckietown_msgs.msg import Segment, SegmentList
from geometry_msgs.msg import Point as PointMsg

from nav_msgs.msg import Odometry
from sensor_msgs.msg import Range
from opencv_apps.msg import FlowArrayStamped, Flow


class OpticalFlowNode(DTROS):
    """
    This node receives the motion vectors from the optical flow algorithm, sends them to be projected to the ground plane,
    and computes the velocity vector from the resulting projected motion vectors.

    Publishers:
        - ~visual_odometry (nav_msgs/Odometry): Estimated velocity vector from the optical flow algorithm, using the projected motion vectors.
        - ~lineseglist_out (duckietown_msgs/SegmentList): The motion vectors as SegmentList
        - ~debug/raw_odometry (nav_msgs/Odometry): The raw odometry vector computed on the un-projected motion vectors

    Subscribers:
        - ~motion_vectors (opencv_apps/FlowArrayStamped): The motion vectors from the optical flow algorithm
        - ~range (sensor_msgs/Range): The range from the range sensor
        - ~projected_motion_vectors (duckietown_msgs/SegmentList): The projected motion vectors from the ground projector.

    """

    def __init__(self, node_name):
        super(OpticalFlowNode, self).__init__(
            node_name=node_name, node_type=NodeType.PERCEPTION
        )

        self.process_frequency = DTParam("~process_frequency", param_type=ParamType.INT)
        self.base_homography_height = DTParam(
            "~base_homography_height", param_type=ParamType.FLOAT
        )

        # TODO: Retrieve the resize scale from the ROS parameter server
        self.resize_scale = 0.0625  # rospy.get_param("~resized/scale_width")
        self.flow_scale = DTParam("~flow_scale", 0.95, param_type=ParamType.FLOAT) # Scales the optical flow from pixels to meters

        # obj attrs
        self.bridge = CvBridge()
        self.last_stamp = rospy.Time.now()

        self._camera_info_initialized = False

        self._range: float = 1.0
        self._scale: float = self._range / self.base_homography_height.value

        self.pub_odometry = rospy.Publisher("~visual_odometry", Odometry, queue_size=1)
        self.pub_seg_list = rospy.Publisher(
            "~lineseglist_out", SegmentList, queue_size=1
        )
        self.pub_debug_raw_odometry = rospy.Publisher(
            "~debug/raw_odometry", Odometry, queue_size=1
        )

        self.sub_motion_vectors = rospy.Subscriber(
            "~motion_vectors", FlowArrayStamped, self.cb_motion_vectors, queue_size=1
        )
        self.sub_range = rospy.Subscriber(
            "~range", Range, self.cb_new_range, queue_size=1
        )
        self.sub_projected_motion_vectors = rospy.Subscriber(
            "~projected_motion_vectors",
            SegmentList,
            self.cb_projected_motion_vectors,
            queue_size=1,
        )

        self.loginfo("Initialization completed.")

    def cb_motion_vectors(self, msg: FlowArrayStamped):
        """
        Callback for the motion vectors from the optical flow algorithm.
        Repack them as SegmentList and publish them to the ground projector.
        """
        segment_list = SegmentList()
        segment_list.header = msg.header

        if msg.flow is None:
            rospy.logwarn("No motion vectors received.")
            return

        segment_list.segments = []

        for flow in msg.flow:
            flow: Flow
            segment = Segment(
                points=[
                    PointMsg(x=flow.point.x, y=flow.point.y),
                    PointMsg(
                        x=flow.point.x + flow.velocity.x * 1,
                        y=flow.point.y + flow.velocity.y * 1,
                    ),
                ]
            )
            segment_list.segments.append(segment)

        if self.pub_debug_raw_odometry.get_num_connections() > 0:
            odometry_msg = Odometry()
            odometry_msg.header = msg.header
            odometry_msg.child_frame_id = "camera"
            odometry_msg.twist.twist.linear.x = np.mean(
                [flow.velocity.x for flow in msg.flow]
            ) * self.flow_scale.value * self._range
            odometry_msg.twist.twist.linear.y = np.mean(
                [flow.velocity.y for flow in msg.flow]
            ) * self.flow_scale.value * self._range

            self.pub_debug_raw_odometry.publish(odometry_msg)

        self.pub_seg_list.publish(segment_list)

    def cb_projected_motion_vectors(self, msg: SegmentList):
        """
        Callback for the motion vectors from the optical flow algorithm.
        It grabs the projected motion vectors and computes the velocity vector using a mean.

        Args:
            msg: the SegmentList message containing the projected motion vectors

        """

        now = rospy.Time.now()

        if msg.segments is None:
            rospy.logwarn_throttle(period=1, msg="Empty motion vectors array received.")
            return

        num_motion_vectors = len(msg.segments)
        motion_vectors = np.zeros((2, num_motion_vectors))

        for i, flow in enumerate(msg.segments):
            flow: Segment
            motion_vectors[:, i] = np.array(
                [
                    flow.points[1].x - flow.points[0].x,
                    flow.points[1].y - flow.points[0].y,
                ]
            )

        velocity = np.mean(np.array(motion_vectors), axis=0)


        # Rotate the velocity vector 90 degrees clockwise to match the odometry frame
        velocity = np.array([velocity[1], velocity[0]])

        assert velocity.shape == (2,), f"Velocity: {velocity}"

        # Publish the optical flow vector as odometry
        odometry_msg = Odometry()
        odometry_msg.header.stamp = now
        odometry_msg.child_frame_id = "base_link"           # TODO: change this to the correct frame
        odometry_msg.twist.twist.linear.x = velocity[0]
        odometry_msg.twist.twist.linear.y = velocity[1]

        self.pub_odometry.publish(odometry_msg)

    def cb_new_range(self, msg: Range):
        self._range = msg.range
        self._scale = self._range / self.base_homography_height.value


if __name__ == "__main__":
    optical_flow_node = OpticalFlowNode("optical_flow_node")
    rospy.spin()
