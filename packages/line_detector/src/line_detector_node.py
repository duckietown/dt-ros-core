#!/usr/bin/env python3
import json
from typing import List, Dict
import numpy as np
import cv2
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import CompressedImage, Image
from duckietown_msgs.msg import Segment as SegmentMsg, SegmentList, AntiInstagramThresholds
from dt_computer_vision.line_detection import LineDetector, ColorRange, Detections
from dt_computer_vision.line_detection.rendering import draw_segments, draw_maps
from image_processing.anti_instagram import AntiInstagram

from duckietown.dtros import DTROS, NodeType, TopicType, DTParam


class LineDetectorNode(DTROS):
    """
    The ``LineDetectorNode`` is responsible for detecting the line white, yellow and red line segment in an image and
    is used for lane localization.

    Upon receiving an image, this node reduces its resolution, cuts off the top part so that only the
    road-containing part of the image is left, extracts the white, red, and yellow segments and publishes them.
    The main functionality of this node is implemented in the :py:class:`line_detector.LineDetector` class.

    The performance of this node can be very sensitive to its configuration parameters. Therefore, it also provides a
    number of debug topics which can be used for fine-tuning these parameters. These configuration parameters can be
    changed dynamically while the node is running via ``rosparam set`` commands.

    Args:
        node_name (:obj:`str`): a unique, descriptive name for the node that ROS will use

    Configuration:
        ~line_detector_parameters (:obj:`dict`): A dictionary with the parameters for the detector. The full list can be found in :py:class:`line_detector.LineDetector`.
        ~colors (:obj:`dict`): A dictionary of colors and color ranges to be detected in the image. The keys (color names) should match the ones in the Segment message definition, otherwise an exception will be thrown! See the ``config`` directory in the node code for the default ranges.
        ~img_size (:obj:`list` of ``int``): The desired downsized resolution of the image. Lower resolution would result in faster detection but lower performance, default is ``[120,160]``
        ~top_cutoff (:obj:`int`): The number of rows to be removed from the top of the image _after_ resizing, default is 40

    Subscriber:
        ~camera_node/image/compressed (:obj:`sensor_msgs.msg.CompressedImage`): The camera images
        ~anti_instagram_node/thresholds(:obj:`duckietown_msgs.msg.AntiInstagramThresholds`): The thresholds to do color correction

    Publishers:
        ~segment_list (:obj:`duckietown_msgs.msg.SegmentList`): A list of the detected segments. Each segment is an :obj:`duckietown_msgs.msg.Segment` message
        ~debug/segments/compressed (:obj:`sensor_msgs.msg.CompressedImage`): Debug topic with the segments drawn on the input image
        ~debug/edges/compressed (:obj:`sensor_msgs.msg.CompressedImage`): Debug topic with the Canny edges drawn on the input image
        ~debug/maps/compressed (:obj:`sensor_msgs.msg.CompressedImage`): Debug topic with the regions falling in each color range drawn on the input image

    """

    def __init__(self, node_name):
        # Initialize the DTROS parent class
        super(LineDetectorNode, self).__init__(node_name=node_name, node_type=NodeType.PERCEPTION)

        # Define parameters
        self._line_detector_parameters = rospy.get_param("~line_detector_parameters", None)
        self._veh = rospy.get_param("~veh")
        self._img_size = rospy.get_param("~img_size", None)
        self._top_cutoff = rospy.get_param("~top_cutoff", None)
        self._colors = DTParam("~colors", None)

        self._traffic_mode = DTParam(f"/{self._veh}/behavior/traffic_mode", None)

        self.bridge = CvBridge()

        # The thresholds to be used for AntiInstagram color correction
        self.ai_thresholds_received = False
        self.anti_instagram_thresholds = dict()
        self.ai = AntiInstagram()

        # This holds the colormaps for the debug/ranges images after they are computed once
        self.colormaps = dict()

        # Create a new LineDetector object with the parameters from the Parameter Server / config file
        self.detector = LineDetector(**self._line_detector_parameters)

        # Update the color ranges objects
        self.color_ranges: Dict[str, ColorRange] = {}
        self.on_colors_range_change()
        self._colors.register_update_callback(self.on_colors_range_change)

        # Publishers
        self.pub_lines = rospy.Publisher(
            "~segment_list", SegmentList, queue_size=1, dt_topic_type=TopicType.PERCEPTION
        )
        self.pub_d_segments = rospy.Publisher(
            "~debug/segments/compressed", CompressedImage, queue_size=1, dt_topic_type=TopicType.DEBUG
        )
        self.pub_d_edges = rospy.Publisher(
            "~debug/edges/compressed", CompressedImage, queue_size=1, dt_topic_type=TopicType.DEBUG
        )
        self.pub_d_maps = rospy.Publisher(
            "~debug/maps/compressed", CompressedImage, queue_size=1, dt_topic_type=TopicType.DEBUG
        )

        # Subscribers
        self.sub_image = rospy.Subscriber(
            "~image/compressed", CompressedImage, self.image_cb, buff_size=10000000, queue_size=1
        )

        self.sub_thresholds = rospy.Subscriber(
            "~thresholds", AntiInstagramThresholds, self.thresholds_cb, queue_size=1
        )

        # Check if CUDA is available
        # LP: I think for this to work we need the base image to have CUDA
        #     which it currently doesn't
        if cv2.cuda.getCudaEnabledDeviceCount() > 0:
            self.loginfo("Using CUDA GPU for line detection.")
            self.cuda_enabled = True
        else:
            self.loginfo("Using the CPU for line detection.")
            self.cuda_enabled = False

    def on_colors_range_change(self):
        self.color_ranges = {
            color: ColorRange.fromDict(d)
            for color, d in list(self._colors.value.items())
        }
        self.loginfo(f"Color range changed to {json.dumps(self._colors.value)}")

    def thresholds_cb(self, thresh_msg):
        self.anti_instagram_thresholds["lower"] = thresh_msg.low
        self.anti_instagram_thresholds["higher"] = thresh_msg.high
        self.ai_thresholds_received = True

    def image_cb(self, image_msg):
        """
        Processes the incoming image messages.

        Performs the following steps for each incoming image:

        #. Performs color correction
        #. Resizes the image to the ``~img_size`` resolution
        #. Removes the top ``~top_cutoff`` rows in order to remove the part of the image that doesn't include the road
        #. Extracts the line segments in the image using :py:class:`line_detector.LineDetector`
        #. Converts the coordinates of detected segments to normalized ones
        #. Creates and publishes the resultant :obj:`duckietown_msgs.msg.SegmentList` message
        #. Creates and publishes debug images if there is a subscriber to the respective topics

        Args:
            image_msg (:obj:`sensor_msgs.msg.CompressedImage`): The receive image message

        """

        # Decode from compressed image with OpenCV
        try:
            obtained_image = self.bridge.compressed_imgmsg_to_cv2(image_msg)
        except ValueError as e:
            self.logerr(f"Could not decode image: {e}")
            return
        
        # Perform color correction
        if self.ai_thresholds_received:
            obtained_image = self.ai.apply_color_balance(
                self.anti_instagram_thresholds["lower"], self.anti_instagram_thresholds["higher"], obtained_image
            )

        if self.cuda_enabled:
            gpu_image = cv2.cuda_GpuMat()
            gpu_image.upload(obtained_image)
        else:
            gpu_image = obtained_image

        # Resize the gpu_image to the desired dimensions
        height_original, width_original = gpu_image.shape[0:2]
        img_size = (self._img_size[1], self._img_size[0])
        if img_size[0] != width_original or img_size[1] != height_original:
            if self.cuda_enabled:
                gpu_image = cv2.cuda.resize(gpu_image, img_size, interpolation=cv2.INTER_NEAREST)
            else:
                gpu_image = cv2.resize(gpu_image, img_size, interpolation=cv2.INTER_NEAREST)

        gpu_image = gpu_image[self._top_cutoff :, :, :]

        # mirror the gpu_image if left-hand traffic mode is set
        if self._traffic_mode.value == "LHT":
            gpu_image = np.fliplr(gpu_image)

        color_order = ["YELLOW", "WHITE", "RED"]
        colors_to_detect = [self.color_ranges[c] for c in color_order]
        # Extract the line segments for every color
        color_detections: List[Detections] = (
            self.detector.detect(gpu_image, colors_to_detect))

        dets: Dict[str, dict] ={}
        for i, detections in enumerate(color_detections):
            color = color_order[i]
            # pack detections in a dictionary
            dets[color] = {
                "lines": detections.lines.tolist(),
                "centers": detections.centers.tolist(),
                "normals": detections.normals.tolist(),
                "color": self.color_ranges[color].representative
            }

        # Construct a SegmentList
        segment_list = SegmentList()
        segment_list.header.stamp = image_msg.header.stamp

        # Remove the offset in coordinates coming from the removing of the top part and
        arr_cutoff = np.array([0, self._top_cutoff, 0, self._top_cutoff])
        arr_ratio = np.array(
            [
                1.0 / self._img_size[1],
                1.0 / self._img_size[0],
                1.0 / self._img_size[1],
                1.0 / self._img_size[0],
            ]
        )

        # Fill in the segment_list with all the detected segments
        for color, det in dets.items():
            # Get the ID for the color from the Segment msg definition
            # Throw and exception otherwise
            if len(det["lines"]) > 0 and len(det["normals"]) > 0:
                try:
                    color_id = getattr(SegmentMsg, color)
                    lines_normalized = (det["lines"] + arr_cutoff) * arr_ratio
                    segment_list.segments.extend(
                        self._to_segment_msg(lines_normalized, det["normals"], color_id)
                    )
                except AttributeError:
                    self.logerr(f"Color name {color} is not defined in the Segment type")

        # Publish the message
        self.pub_lines.publish(segment_list)
        
        if self.cuda_enabled:
            # Download the image from gpu memory
            image = gpu_image.download()
        else:
            # Just rename appropriately the image variable
            image = gpu_image

        # If there are any subscribers to the debug topics, generate a debug image and publish it
        if self.pub_d_segments.get_num_connections() > 0:
            debug_img = draw_segments(image,
                                      {
                                        self.color_ranges["YELLOW"]: color_detections[0],
                                        self.color_ranges["WHITE"]: color_detections[1],
                                        self.color_ranges["RED"]: color_detections[2]
                                      }
                                    )

            # mirror the image if left-hand traffic mode is set
            if self._traffic_mode.value == "LHT":
                debug_img = np.fliplr(debug_img)
            debug_image_msg = self.bridge.cv2_to_compressed_imgmsg(debug_img)
            debug_image_msg.header = image_msg.header
            self.pub_d_segments.publish(debug_image_msg)

        if self.pub_d_edges.get_num_connections() > 0:
            canny_edges = self.detector.find_edges(image,
                                                   self.detector.canny_thresholds[0],
                                                   self.detector.canny_thresholds[1],
                                                   self.detector.canny_aperture_size
                                                   )
            # mirror the image if left-hand traffic mode is set
            if self._traffic_mode.value == "LHT":
                canny_edges = np.fliplr(canny_edges)
            debug_image_msg = self.bridge.cv2_to_compressed_imgmsg(canny_edges)
            debug_image_msg.header = image_msg.header
            self.pub_d_edges.publish(debug_image_msg)

        if self.pub_d_maps.get_num_connections() > 0:
#            colorrange_detections = {self.color_ranges[c]: det for c, det in list(detections.items())}
            debug_img = draw_maps(image,
                                  {
                                      self.color_ranges["YELLOW"]: color_detections[0],
                                      self.color_ranges["WHITE"]: color_detections[1],
                                      self.color_ranges["RED"]: color_detections[2]
                                  }
                                  )
            # mirror the image if left-hand traffic mode is set
            if self._traffic_mode.value == "LHT":
                debug_img = np.fliplr(debug_img)
            debug_image_msg = self.bridge.cv2_to_compressed_imgmsg(debug_img)
            debug_image_msg.header = image_msg.header
            self.pub_d_maps.publish(debug_image_msg)


    @staticmethod
    def _to_segment_msg(lines, normals, color):
        """
        Converts line detections to a list of Segment messages.

        Converts the resultant line segments and normals from the line detection to a list of Segment messages.

        Args:
            lines (:obj:`numpy array`): An ``Nx4`` array where each row represents a line.
            normals (:obj:`numpy array`): An ``Nx2`` array where each row represents the normal of a line.
            color (:obj:`str`): Color name string, should be one of the pre-defined in the Segment message definition.

        Returns:
            :obj:`list` of :obj:`duckietown_msgs.msg.Segment`: List of Segment messages

        """
        segment_msg_list = []
        for x1, y1, x2, y2, norm_x, norm_y in np.hstack((lines, normals)):
            segment = SegmentMsg()
            segment.color = color
            segment.pixels_normalized[0].x = x1
            segment.pixels_normalized[0].y = y1
            segment.pixels_normalized[1].x = x2
            segment.pixels_normalized[1].y = y2
            segment.normal.x = norm_x
            segment.normal.y = norm_y
            segment_msg_list.append(segment)
        return segment_msg_list

    def _plot_ranges_histogram(self, channels):
        """Utility method for plotting color histograms and color ranges.

        Args:
            channels (:obj:`str`): The desired two channels, should be one of ``['HS','SV','HV']``

        Returns:
            :obj:`numpy array`: The resultant plot image

        """
        channel_to_axis = {"H": 0, "S": 1, "V": 2}
        axis_to_range = {0: 180, 1: 256, 2: 256}

        # Get which is the third channel that will not be shown in this plot
        missing_channel = "HSV".replace(channels[0], "").replace(channels[1], "")

        hsv_im = self.detector.hsv
        # Get the pixels as a list (flatten the horizontal and vertical dimensions)
        hsv_im = hsv_im.reshape((-1, 3))

        channel_idx = [channel_to_axis[channels[0]], channel_to_axis[channels[1]]]

        # Get only the relevant channels
        x_bins = np.arange(0, axis_to_range[channel_idx[1]] + 1, 2)
        y_bins = np.arange(0, axis_to_range[channel_idx[0]] + 1, 2)
        h, _, _ = np.histogram2d(
            x=hsv_im[:, channel_idx[0]], y=hsv_im[:, channel_idx[1]], bins=[y_bins, x_bins]
        )
        # Log-normalized histogram
        np.log(h, out=h, where=(h != 0))
        h = (255 * h / np.max(h)).astype(np.uint8)

        # Make a color map, for the missing channel, just take the middle of the range
        if channels not in self.colormaps:
            colormap_1, colormap_0 = np.meshgrid(x_bins[:-1], y_bins[:-1])
            colormap_2 = np.ones_like(colormap_0) * (axis_to_range[channel_to_axis[missing_channel]] / 2)

            channel_to_map = {channels[0]: colormap_0, channels[1]: colormap_1, missing_channel: colormap_2}

            self.colormaps[channels] = np.stack(
                [channel_to_map["H"], channel_to_map["S"], channel_to_map["V"]], axis=-1
            ).astype(np.uint8)

            if self.cuda_enabled:
                self.colormaps[channels] = cv2.cuda.cvtColor(self.colormaps[channels], cv2.COLOR_HSV2BGR)
            else:
                self.colormaps[channels] = cv2.cvtColor(self.colormaps[channels], cv2.COLOR_HSV2BGR)

        # resulting histogram image as a blend of the two images
        if self.cuda_enabled:
            im = cv2.cuda.cvtColor(h[:, :, None], cv2.COLOR_GRAY2BGR)
        else:
            im = cv2.cvtColor(h[:, :, None], cv2.COLOR_GRAY2BGR)
            
        im = cv2.addWeighted(im, 0.5, self.colormaps[channels], 1 - 0.5, 0.0)

        # now plot the color ranges on top
        for _, color_range in list(self.color_ranges.items()):
            # convert HSV color to BGR
            c = color_range.representative
            c = np.uint8([[[c[0], c[1], c[2]]]])
            if self.cuda_enabled:
                color = cv2.cuda.cvtColor(c, cv2.COLOR_HSV2BGR).squeeze().astype(int).tolist()
            else:
                color = cv2.cvtColor(c, cv2.COLOR_HSV2BGR).squeeze().astype(int).tolist()

            for i in range(len(color_range.low)):
                cv2.rectangle(
                    im,
                    pt1=(
                        (color_range.high[i, channel_idx[1]] / 2).astype(np.uint8),
                        (color_range.high[i, channel_idx[0]] / 2).astype(np.uint8),
                    ),
                    pt2=(
                        (color_range.low[i, channel_idx[1]] / 2).astype(np.uint8),
                        (color_range.low[i, channel_idx[0]] / 2).astype(np.uint8),
                    ),
                    color=color,
                    lineType=cv2.LINE_4,
                )
        # ---
        return im


if __name__ == "__main__":
    # Initialize the node
    line_detector_node = LineDetectorNode(node_name="line_detector_node")
    # Keep it spinning to keep the node alive
    rospy.spin()
