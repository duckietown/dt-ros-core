#!/usr/bin/env python3

import os
from typing import Optional, Union

from dt_computer_vision.camera import Pixel
import numpy as np
from dt_computer_vision.camera.types import NormalizedImagePoint
from dt_computer_vision.ground_projection import GroundPoint
import rospy
from cv_bridge import CvBridge
from duckietown_msgs.msg import Segment, SegmentList
from sensor_msgs.msg import CameraInfo, CompressedImage
from geometry_msgs.msg import Point as PointMsg

from dt_computer_vision.camera import CameraModel
from dt_computer_vision.ground_projection import GroundProjector

from dt_computer_vision.camera.homography import Homography, HomographyToolkit
from duckietown.dtros import DTROS, NodeType

# FIXME: Is this still used?

# @dataclass
# class ImageProjectorConfig:
#     roi : RegionOfInterest
#     ppm : int


class GroundProjectionNode(DTROS):
    """
    This node projects the line segments detected in the image to the ground plane and in the robot's
    reference frame.
    In this way it enables lane localization in the 2D ground plane. This projection is performed using the
    homography
    matrix obtained from the extrinsic calibration procedure.

    Args:
        node_name (:obj:`str`): a unique, descriptive name for the node that ROS will use

    Subscribers:
        ~camera_info (:obj:`sensor_msgs.msg.CameraInfo`): Intrinsic properties of the camera. Needed for
        rectifying the segments.
        ~lineseglist_in (:obj:`duckietown_msgs.msg.SegmentList`): Line segments in pixel space from
        unrectified images
        ~image/compressed (:obj:`sensor_msgs.msg.CompressedImage`): Compressed image from the camera.
        Only needed if you want to visualize the debug image (i.e. the image with the homography applied).

    Publishers:
        ~lineseglist_out (:obj:`duckietown_msgs.msg.SegmentList`): Line segments in the ground plane
        relative to the robot origin
        ~debug/ground_projection_image/compressed (:obj:`sensor_msgs.msg.CompressedImage`): Debug image
        that shows the robot relative to the projected segments. Useful to check if the extrinsic
        calibration is accurate.
        ~debug/projected_image/rectified/compressed (:obj:`sensor_msgs.msg.CompressedImage`): Debug image
        that shows the rectified image. Useful to check if the rectification is accurate.
        ~debug/projected_image/compressed (:obj:`sensor_msgs.msg.CompressedImage`): Debug image that shows
        the projected image. Useful to check if the homography is accurate.
    """

    bridge: CvBridge
    projector: Optional[GroundProjector] = None

    def __init__(self, node_name: str):
        # Initialize the DTROS parent class
        super(GroundProjectionNode, self).__init__(
            node_name=node_name, node_type=NodeType.PERCEPTION, fsm_controlled=True
        )

        self.bridge = CvBridge()
        self.projector: Optional[GroundProjector] = None
        self.camera: Optional[CameraModel] = None
        self.homography: Optional[Homography] = None
        self._first_processing_done = False
        self.camera_info_received = False

        # FIXME: Is this still used?
        # self._image_projector_config: dict = rospy.get_param(
        #     "~image_projector_configuration", None, type=dict
        # )

        # subscribers
        self.sub_camera_info = rospy.Subscriber(
            "~camera_info", CameraInfo, self.cb_camera_info, queue_size=1
        )
        self.sub_lineseglist_ = rospy.Subscriber(
            "~lineseglist_in", SegmentList, self.lineseglist_cb, queue_size=1
        )

        # publishers
        self.pub_lineseglist = rospy.Publisher(
            "~lineseglist_out", SegmentList, queue_size=1
        )
        self.pub_debug_road_view_img = rospy.Publisher(
            "~debug/ground_projection_image/compressed",
            CompressedImage,
            queue_size=1,
        )

        self.pub_debug_rectified_img = rospy.Publisher(
            "~debug/projected_image/rectified/compressed",
            CompressedImage,
            queue_size=1,
        )

        self.pub_debug_projected_img = rospy.Publisher(
            "~debug/projected_image/compressed",
            CompressedImage,
            queue_size=1,
        )

        self.bridge = CvBridge()

        self.debug_img_bg = None

        # Seems to be never used:
        # TODO: what was the intent of these services? They seem to be redundant if we have a proper tf tree
        # self.service_gnd_coord_ = rospy.Service("~get_ground_coordinate", GetGroundCoord,
        # self.get_ground_coordinate_cb)
        # self.service_img_coord_ = rospy.Service("~get_image_coordinate", GetImageCoord,
        # self.get_image_coordinate_cb)

    def cb_camera_info(self, msg: CameraInfo):
        """
        Initializes a :py:class:`image_processing.GroundProjectionGeometry` object and a
        :py:class:`image_processing.Rectify` object for image rectification

        Args:
            msg (:obj:`sensor_msgs.msg.CameraInfo`): Intrinsic properties of the camera.

        """
        if not self.camera_info_received:
            self.log("Received camera info message")
            # create camera object
            self.camera = CameraModel(
                width=msg.width,
                height=msg.height,
                K=np.reshape(msg.K, (3, 3)),
                D=np.reshape(msg.D, (5,)),
                P=np.reshape(msg.P, (3, 4)),
            )

            self.homography = self.load_extrinsics()
            self.camera.H = self.homography
            self.projector = GroundProjector(self.camera)

            self.loginfo("Camera model initialized")

        self.camera_info_received = True

    def _pixel_to_ground(self, p: Pixel) -> GroundPoint:
        """
        Converts a pixel coordinate to a ground point.

        Args:
            p (:obj:`dt_computer_vision.camera.Pixel`): Pixel coordinate

        Returns:
            :obj:`dt_computer_vision.ground_projection`: Ground point

        """

        # TODO: check if we need to rectify the pixels
        if self.camera is None:
            raise ValueError("Camera model not initialized")

        # rectified pixel to normalized coordinates
        p_norm: NormalizedImagePoint = self.camera.pixel2vector(p)

        # project image point onto the ground plane
        p_ground: GroundPoint = self.projector.vector2ground(p_norm)

        return p_ground

    def pixel_msg_to_ground_msg(self, pixel_msg: PointMsg) -> PointMsg:
        """
        Converts a pixel message to a ground message.

        Args:
            pixel_msg (:obj:`geometry_msgs.msg.Point`): Pixel message

        Returns:
            :obj:`geometry_msgs.msg.Point`: Ground message

        """
        p = Pixel(x=pixel_msg.x, y=pixel_msg.y)
        p_ground = self._pixel_to_ground(p)
        return PointMsg(x=p_ground.x, y=p_ground.y)

    def lineseglist_cb(self, seglist_msg: SegmentList):
        """
        Projects a list of line segments on the ground reference frame point by point by
        calling :py:meth:`pixel_msg_to_ground_msg`. Then publishes the projected list of segments.

        Args:
            seglist_msg (:obj:`duckietown_msgs.msg.SegmentList`): Line segments in pixel space from
            unrectified images

        """
        if self.camera_info_received:
            seglist_out = SegmentList()
            seglist_out.header = seglist_msg.header
            for received_segment in seglist_msg.segments:
                received_segment: Segment
                projected_segment = Segment()
                projected_segment.points[0] = self.pixel_msg_to_ground_msg(
                    received_segment.points[0]
                )
                projected_segment.points[1] = self.pixel_msg_to_ground_msg(
                    received_segment.points[1]
                )
                projected_segment.color = received_segment.color
                # TODO what about normal?
                seglist_out.segments.append(projected_segment)
            self.pub_lineseglist.publish(seglist_out)

            if not self._first_processing_done:
                self.log("First projected segments published.")
                self._first_processing_done = True

            if self.pub_debug_road_view_img.get_num_connections() > 0:
                return  # TODO: Reimplement using debug_image from dt_computer_vision
                debug_image_msg = self.bridge.cv2_to_compressed_imgmsg(
                    self.debug_image(seglist_out)
                )
                debug_image_msg.header = seglist_out.header
                self.pub_debug_road_view_img.publish(debug_image_msg)
        else:
            self.log("Waiting for a CameraInfo message", "warn")

    def load_extrinsics(self) -> Union[Homography, None]:
        """
        Loads the homography matrix from the extrinsic calibration file.

        Returns:
            :obj:`Homography`: the loaded homography matrix

        """
        # load extrinsic calibration
        cali_file_folder = "/data/config/calibrations/camera_extrinsic/"
        cali_file = cali_file_folder + rospy.get_namespace().strip("/") + ".yaml"

        # Locate calibration yaml file or use the default otherwise
        if not os.path.isfile(cali_file):
            self.log(
                f"Can't find calibration file: {cali_file}\n Using default calibration instead.",
                "warn",
            )
            cali_file = os.path.join(cali_file_folder, "default.yaml")

        # Shutdown if no calibration file not found
        if not os.path.isfile(cali_file):
            msg = "Found no calibration file ... aborting"
            self.logerr(msg)
            rospy.signal_shutdown(msg)

        try:
            H: Homography = HomographyToolkit.load_from_disk(
                cali_file, return_date=False
            )  # type: ignore
            return H.reshape((3, 3))
        except Exception as e:
            msg = f"Error in parsing calibration file {cali_file}:\n{e}"
            self.logerr(msg)
            rospy.signal_shutdown(msg)


if __name__ == "__main__":
    ground_projection_node = GroundProjectionNode(node_name="ground_projection_node")
    rospy.spin()
