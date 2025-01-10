#!/usr/bin/env python3

import os
import numpy as np
import cv2
import rospy
from duckietown_msgs.msg import LEDPattern, ObstacleProjectedDetectionList
from duckietown_msgs.srv import SetCustomLEDPattern, SetCustomLEDPatternRequest
from sensor_msgs.msg import CompressedImage, Image
from duckietown.dtros import DTROS, NodeType
from std_msgs.msg import String

class ObjectDetectorDebugNode(DTROS):
    FILE_NAME = "/data/object-detection-log-debug.mp4"
    FRAME_NB = 200

    def __init__(self, node_name):
        # initialize the DTROS parent class
        super(ObjectDetectorDebugNode, self).__init__(node_name=node_name, node_type=NodeType.GENERIC)
        self.veh = rospy.get_namespace().strip("/")
        # construct publishers | subscribers | services
        # self.sub = rospy.Subscriber(f"/{self.veh}/object_detection/image/debug", Image, self.callback)
        self.sub = rospy.Subscriber(f"object_detection/image/compressed", CompressedImage, self.callback)
        self.obs_detect_info = rospy.Subscriber(f"object_detection/detections", ObstacleProjectedDetectionList, self.read_topic_detection)
        self.is_first = True
        self.on_finish = True
        self.buffer = []

    def run(self):
        return

    def read_topic_detection(self, data):
        rospy.loginfo(f"Subscriber testing DETECT OBSTACLES INFO, {data}")

    def callback(self, data):
        if len(self.buffer) < self.FRAME_NB:
            rospy.loginfo(f"frame {len(self.buffer)}")
            self.buffer.append(cv2.imdecode(np.frombuffer(data.data, np.uint8), cv2.IMREAD_COLOR))

        elif self.on_finish:
            height, width, layer = self.buffer[0].shape
            size = (width, height)
            
            out = cv2.VideoWriter(self.FILE_NAME, cv2.VideoWriter_fourcc(*'MP4V'), 30, size)

            for i in range(len(self.buffer)):
                out.write(self.buffer[i])
            out.release()

            rospy.loginfo("finished!")
            rospy.loginfo(os.path.abspath(self.FILE_NAME))
            self.on_finish = False


if __name__ == '__main__':
    # create the node
    node = ObjectDetectorDebugNode(node_name='object_detector_debug_node')
    # keep spinning
    rospy.spin()