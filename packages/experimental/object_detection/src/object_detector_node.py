#!/usr/bin/env python3

import os
import cv2
import rospy
import numpy as np
from duckietown.dtros import DTROS, NodeType, TopicType
from std_msgs.msg import String
from sensor_msgs.msg import CompressedImage, Image
from geometry_msgs.msg import Point
from duckietown_msgs.msg import ObstacleProjectedDetectionList, ObstacleProjectedDetection, ObstacleType
from cv_bridge import CvBridge
from dt_device_utils import DeviceHardwareBrand, get_device_hardware_brand
from object_detection import ModelWrapper, find_position


SKIPED_FRAME = 5

class ObjectDetectorNode(DTROS):

    def __init__(self, node_name, model_name="simple_model"):
        #* initialize the DTROS parent class
        super(ObjectDetectorNode, self).__init__(node_name=node_name, node_type=NodeType.GENERIC)
        self.veh = rospy.get_namespace().strip("/")
        self.initialized = False
        #* construct publishers 
        self.pub_detections_image = rospy.Publisher("~image/compressed", CompressedImage, queue_size=1, dt_topic_type=TopicType.DEBUG)
        self.pub_obs_list = rospy.Publisher("~detections", ObstacleProjectedDetectionList, queue_size=1)

        #* Subscribe to the image
        self.sub_img = rospy.Subscriber(
            f"/{self.veh}/camera_node/image/compressed",
            CompressedImage,
            self.callback,
            buff_size=10000000,
            queue_size=1
        )

        #* Params
        self.bridge = CvBridge()
        self.debug = True
        # self.model_name = "baseline"
        self.model_name = "yolov5n"
        self.dt_token = "dt1-3nT8KSoxVh4Migd7N6Nsjy5q8BHtzjcsyz57x9FyJbx5UhJ-43dzqWFnWd8KBa1yev1g3UKnzVxZkkTbfex5eXnmoSTSmB3YdtDmc5tQuXNDk3cQ74"
        rospy.loginfo(f"Model name: {self.model_name}")

        self.model = ModelWrapper(self.model_name, self.dt_token)

        self.frame_id = 0
        self.initialized = True
        self.first_run = True
    

    def callback(self, img_msg):
        if not self.initialized:
            return

        if self.frame_id < SKIPED_FRAME:
            self.frame_id += 1
            return
        else:
            self.frame_id = 0

        rospy.loginfo(f"Processing image")

        #* img: ROS -> OpenCV 
        try:
            image = self.bridge.compressed_imgmsg_to_cv2(img_msg)
        except ValueError as e:
            self.logerr(f"[obj-detector] Could not convert img_msg to img: {e}")
            return

        #* Resize img and color
        old_img = image
        #* Handle sim
        if get_device_hardware_brand() != DeviceHardwareBrand.JETSON_NANO:  
            image = image[...,::-1].copy()  # image is bgr, flip it to rgb

        img_size = (416,416)
        old_img = cv2.resize(old_img, img_size)
        image = cv2.resize(image, img_size)

        #* Do prediction from the img
        bboxes, classes, scores = self.model.infer(image)

        #* Find position of the objects detected
        positions = find_position(bboxes, img_size[0], img_size[1])

        #* Publish detected obstacle
        obs_list = ObstacleProjectedDetectionList()
        obs_list.list = list()
        for box_points, clas in zip(positions, classes):
            cnt, points = box_points
            obsType = ObstacleType()
            obsType.type = int(clas)
            obs = ObstacleProjectedDetection()
            ptx_mean = points[0].x + points[1].x
            pty_mean = points[0].y + points[1].y
            pt_mean = Point()
            pt_mean.x = ptx_mean
            pt_mean.y = pty_mean
            pt_mean.z = 0
            obs.location = pt_mean
            obs.type = obsType
            obs.distance = np.sqrt(obs.location.x**2 + obs.location.y**2)
            obs_list.list.append(obs)
        if len(obs_list.list) != 0:
            self.pub_obs_list.publish(obs_list)

        # #* Add bounding box
        # for cnt, points in positions:
        #     x,y,w,h = cv2.boundingRect(cnt)
        #     old_img = cv2.rectangle(old_img,(x,y),(x+w,y+h),(255,10,10),2)
        #     x_1, y_1 = round(points[0].x, 3), round(points[0].y, 3)
        #     x_2, y_2 = round(points[1].x, 3), round(points[1].y, 3)
        #     # rospy.loginfo(f"point 1: ({x_1}, {y_1}) | point 2: ({x_2}, {y_2})")

        #     old_img = cv2.putText(old_img, f"({x_1}, {y_1})m | ({x_2}, {y_2})m", (x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255,10,10),2)

        if self.debug:
            colors = {0: (0, 255, 255), 1: (0, 165, 255), 2: (0, 250, 0), 3: (0, 0, 255)}
            names = {0: "duckie", 1: "duckiebot", 2: "truck", 3: "bus"}
            font = cv2.FONT_HERSHEY_SIMPLEX
            for clas, box in zip(classes, bboxes):

                rospy.loginfo(f"Detected: {clas} at: {box}")

                pt1 = np.array([int(box[0]), int(box[1])])
                pt2 = np.array([int(box[2]), int(box[3])])
                pt1 = tuple(pt1)
                pt2 = tuple(pt2)
                color = colors[int(clas)]
                name = names[int(clas)]
                old_img = cv2.rectangle(old_img, pt1, pt2, color, 2)
                text_location = (pt1[0], min(416, pt1[1]+20))
                old_img = cv2.putText(old_img, name, text_location, font, 1, color, thickness=3)

        #* Publish detection img for demo
        obj_det_img = self.bridge.cv2_to_compressed_imgmsg(old_img)
        self.pub_detections_image.publish(obj_det_img)


    def run(self):
        # publish message every 1 second
        if self.first_run:
            message = f"{os.environ['VEHICLE_NAME']} is ready for take-off! 3, 2, 1, Brrrrrrrr"
            rospy.loginfo(f"[obj-detector] Publishing message: {message}")

if __name__ == '__main__':
    # create the node
    node = ObjectDetectorNode(node_name='object_detector_node')
    # run node
    node.run()
    # keep spinning
    rospy.spin()