#!/usr/bin/env python3
import rospkg
import rospy
import yaml
from duckietown_msgs.msg import (
    AprilTagsWithInfos,
    TagInfo,
    BoolStamped,
    AprilTagDetection,
    AprilTagDetectionArray,
)
import numpy as np
import tf.transformations as tr
from geometry_msgs.msg import PoseStamped, Pose
from std_msgs.msg import Header

from duckietown.dtros import DTROS, NodeType, TopicType, DTParam, ParamType

class AprilPostPros(DTROS):
    """ """

    def __init__(self, node_name):
        """ """
        super(AprilPostPros, self).__init__(
            node_name=node_name,
            node_type=NodeType.PERCEPTION,
            fsm_controlled=True
        )
        self.node_name = node_name

        # Load parameters
        self.camera_x = self.setupParam("~camera_x", 0.065)
        self.camera_y = self.setupParam("~camera_y", 0.0)
        self.camera_z = self.setupParam("~camera_z", 0.11)
        self.camera_theta = self.setupParam("~camera_theta", 19.0)
        self.scale_x = self.setupParam("~scale_x", 1)
        self.scale_y = self.setupParam("~scale_y", 1)
        self.scale_z = self.setupParam("~scale_z", 1)

        # -------- Start adding back the tag info stuff

        tags_filepath = self.setupParam("~tags_file")

        self.loc = self.setupParam("~loc", -1)  # -1 if no location is given
        tags_file = open(tags_filepath, "r")
        self.tags_dict = yaml.safe_load(tags_file)
        tags_file.close()
        self.info = TagInfo()

        self.sign_types = {
            "StreetName": self.info.S_NAME,
            "TrafficSign": self.info.SIGN,
            "Light": self.info.LIGHT,
            "Localization": self.info.LOCALIZE,
            "Vehicle": self.info.VEHICLE,
        }
        self.traffic_sign_types = {
            "stop": self.info.STOP,
            "yield": self.info.YIELD,
            "no-right-turn": self.info.NO_RIGHT_TURN,
            "no-left-turn": self.info.NO_LEFT_TURN,
            "oneway-right": self.info.ONEWAY_RIGHT,
            "oneway-left": self.info.ONEWAY_LEFT,
            "4-way-intersect": self.info.FOUR_WAY,
            "right-T-intersect": self.info.RIGHT_T_INTERSECT,
            "left-T-intersect": self.info.LEFT_T_INTERSECT,
            "T-intersection": self.info.T_INTERSECTION,
            "do-not-enter": self.info.DO_NOT_ENTER,
            "pedestrian": self.info.PEDESTRIAN,
            "t-light-ahead": self.info.T_LIGHT_AHEAD,
            "duck-crossing": self.info.DUCK_CROSSING,
            "parking": self.info.PARKING,
        }

        # ---- end tag info stuff

        self.sub_prePros = rospy.Subscriber(
            "~detections", AprilTagDetectionArray, self.callback, queue_size=1
        )
        self.pub_postPros = rospy.Publisher("~apriltags_out", AprilTagsWithInfos, queue_size=1)
        self.pub_visualize = rospy.Publisher("~tag_pose", PoseStamped, queue_size=1)


        rospy.loginfo("[%s] has started", self.node_name)

    def setupParam(self, param_name, default_value=rospy.client._Unspecified):
        value = rospy.get_param(param_name, default_value)
        rospy.set_param(param_name, value)
        return value

    def callback(self, msg):

        tag_infos = []

        new_tag_data = AprilTagsWithInfos()

        # Load tag detections message
        for detection in msg.detections:

            # ------ start tag info processing

            new_info = TagInfo()
            # Can use id 1 as long as no bundles are used
            new_info.id = int(detection.tag_id)
            id_info = self.tags_dict[new_info.id]

            # Check yaml file to fill in ID-specific information
            new_info.tag_type = self.sign_types[id_info["tag_type"]]
            if new_info.tag_type == self.info.S_NAME:
                new_info.street_name = id_info["street_name"]
            elif new_info.tag_type == self.info.SIGN:
                new_info.traffic_sign_type = self.traffic_sign_types[id_info["traffic_sign_type"]]
            elif new_info.tag_type == self.info.VEHICLE:
                new_info.vehicle_name = id_info["vehicle_name"]

            tag_infos.append(new_info)
            # --- end tag info processing

            # Define the transforms
            veh_t_camxout = tr.translation_matrix((self.camera_x, self.camera_y, self.camera_z))
            veh_R_camxout = tr.euler_matrix(0, self.camera_theta * np.pi / 180, 0, "rxyz")
            veh_T_camxout = tr.concatenate_matrices(
                veh_t_camxout, veh_R_camxout
            )  # 4x4 Homogeneous Transform Matrix

            camxout_T_camzout = tr.euler_matrix(-np.pi / 2, 0, -np.pi / 2, "rzyx")
            veh_T_camzout = tr.concatenate_matrices(veh_T_camxout, camxout_T_camzout)

            tagzout_T_tagxout = tr.euler_matrix(-np.pi / 2, 0, np.pi / 2, "rxyz")

            # Load translation
            trans = detection.transform.translation
            rot = detection.transform.rotation
            camzout_t_tagzout = tr.translation_matrix(
                (trans.x * self.scale_x, trans.y * self.scale_y, trans.z * self.scale_z)
            )
            camzout_R_tagzout = tr.quaternion_matrix((rot.x, rot.y, rot.z, rot.w))
            camzout_T_tagzout = tr.concatenate_matrices(camzout_t_tagzout, camzout_R_tagzout)

            veh_T_tagxout = tr.concatenate_matrices(veh_T_camzout, camzout_T_tagzout, tagzout_T_tagxout)

            # Overwrite transformed value
            (trans.x, trans.y, trans.z) = tr.translation_from_matrix(veh_T_tagxout)
            (rot.x, rot.y, rot.z, rot.w) = tr.quaternion_from_matrix(veh_T_tagxout)

            new_tag_data.detections.append(detection)

        new_tag_data.infos = tag_infos
        # Publish Message
        self.pub_postPros.publish(new_tag_data)


if __name__ == "__main__":
    # rospy.init_node("AprilPostPros", anonymous=False)
    node = AprilPostPros(node_name="apriltag_postprocessing_node")
    rospy.spin()
