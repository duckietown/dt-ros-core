#!/usr/bin/env python3

import rospy
from BaseComNode import BaseComNode
from UtilStates import ActionState, IntersectionType
from duckietown.dtros import DTROS, NodeType, TopicType
from sensor_msgs.msg import CompressedImage
from duckietown_msgs.msg import LEDPattern, BoolStamped, Twist2DStamped, AprilTagsWithInfos
from duckietown_msgs.srv import SetCustomLEDPattern, SetCustomLEDPatternRequest, ChangePattern
from duckietown.dtros.utils import apply_namespace
from std_msgs.msg import String
import time
from duckietown_msgs.msg import TagInfo # For tests simulating intersection type


class CommunicationNode(DTROS, BaseComNode):

    def __init__(self, node_name):
        # Initialize the DTROS parent class
        DTROS.__init__(self, node_name=node_name, node_type=NodeType.COMMUNICATION)

        # Init parameters dictionary and fill using the node parameters
        params = dict()
        # BaseComNode
        params["~time_out_sec"] = rospy.get_param("~time_out_sec", None)
        params["~fps_history_duration_sec"] = rospy.get_param("~fps_history_duration_sec", None)
        params["~fps_update_period_sec"] = rospy.get_param("~fps_update_period_sec", None)
        params["~default_img_fps"] = rospy.get_param("~default_img_fps", None)
        # Stop Sign Solver
        params["~permitted_freq_list"] = rospy.get_param("~permitted_freq_list", None)
        params["~freq_error_upper_margin"] = rospy.get_param("~freq_error_upper_margin", None)
        params["~sensing_duration_sec"] = rospy.get_param("~sensing_duration_sec", None)
        params["~ss_buffer_length"] = rospy.get_param("~ss_buffer_length", None)
        params["~ss_buffer_forget_time"] = rospy.get_param("~ss_buffer_forget_time", None)
        params["~ss_brightness_threshold"] = rospy.get_param("~ss_brightness_threshold", None)
        params["~ss_max_diff_threshold"] = rospy.get_param("~ss_max_diff_threshold", None)
        params["~ss_g_blur_sigma_x"] = rospy.get_param("~ss_g_blur_sigma_x", None)
        params["~ss_g_blur_sigma_y"] = rospy.get_param("~ss_g_blur_sigma_y", None)
        # Traffic Light Solver
        params["~tl_green_blink_freq"] = rospy.get_param("~tl_green_blink_freq", None)
        params["~tl_tolerance_freq_diff"] = rospy.get_param("~tl_tolerance_freq_diff", None)
        params["~max_traffic_light_height"] = rospy.get_param("~max_traffic_light_height", None)
        params["~tl_buffer_length"] = rospy.get_param("~tl_buffer_length", None)
        params["~tl_buffer_forget_time"] = rospy.get_param("~tl_buffer_forget_time", None)
        params["~tl_brightness_threshold"] = rospy.get_param("~tl_brightness_threshold", None)
        params["~tl_max_diff_threshold"] = rospy.get_param("~tl_max_diff_threshold", None)
        params["~tl_g_blur_sigma_x"] = rospy.get_param("~tl_g_blur_sigma_x", None)
        params["~tl_g_blur_sigma_y"] = rospy.get_param("~tl_g_blur_sigma_y", None)
        # Debug/demo mode
        params["~ss_demo_mode"] = rospy.get_param("~ss_demo_mode", None)
        params["~tl_demo_mode"] = rospy.get_param("~tl_demo_mode", None)

        # Initialize the BaseComNode parent class
        # Note: self.params will be set here
        BaseComNode.__init__(self, params)


        # Subscribing
        self.img_sub = rospy.Subscriber('~image_in', CompressedImage, self.img_callback)
        # Intersection_type subscriber is initialized in on_switch_on(self), this way we ensure we only subscribe and unregister (just before Go is published)
        # when the node is started and stopped. This is important so we avoid picking wrong april tags and keep running when our node is disconnected
        self.intersectype_sub = None

        # Publishing
        self.pub_intersection_go = rospy.Publisher("~intersection_go", BoolStamped, queue_size=1)
        self.pub_timed_out = rospy.Publisher("~timed_out", BoolStamped, queue_size=1)
        #self.pub_coord_cmd = rospy.Publisher("~car_cmd", Twist2DStamped, queue_size=1, dt_topic_type=TopicType.CONTROL)

        # Used Services
        self.changeCustomPattern = rospy.ServiceProxy(
            '~set_custom_pattern',
            SetCustomLEDPattern
        )

        # TODO Debug/Demo testing:
        if self.params["~tl_demo_mode"] or self.params["~ss_demo_mode"]:
            time.sleep(5) # Wait for LED emitter to be running.
            new_info = TagInfo()
            new_info.tag_type = new_info.SIGN
            new_tag_data = AprilTagsWithInfos()        
            # simulate stop sign
            if self.params["~ss_demo_mode"]:
                new_info.traffic_sign_type = new_info.STOP
                new_tag_data.infos.append(new_info)
                self.intersection_type_callback(new_tag_data)
            # simulate TL
            if self.params["~tl_demo_mode"]:
                new_info.traffic_sign_type = new_info.T_LIGHT_AHEAD
                new_tag_data.infos.append(new_info)
                self.intersection_type_callback(new_tag_data)

    def on_switch_on(self):
        # Intersection_type subscriber is initialized in on_switch_on(self), this way we ensure we only subscribe and unregister (just before Go is published)
        # when the node is started and stopped. This is important so we avoid picking wrong april tags and keep running when our node is disconnected
        self.intersectype_sub = rospy.Subscriber('~intersection_type_in', AprilTagsWithInfos, self.intersection_type_callback,queue_size=1)

    def on_switch_off(self):
        print("on_switch_off")

    def blink_at(self, frequency: int = 0, color: str='white'):

        BaseComNode.blink_at(self, frequency, color)

        # Build LED message
        pattern_msg = LEDPattern(
            frequency=frequency,
            color_list=[color] * 5,
            color_mask=[1] * 5,
            frequency_mask=[1] * 5,
        )
        self.changeCustomPattern(pattern_msg)

    def publish_signal(self, action):
        message = BoolStamped()
        message.header.stamp = rospy.Time.now()

        if action is ActionState.Go:
            # No more needed for the tag information -> unsubscribe
            if self.intersectype_sub:    
                self.intersectype_sub.unregister()

            # Set color to solid Green for 1 sec
            self.blink_at(frequency=0, color='green')
            time.sleep(2)
            # Trun the LEDs off
            self.blink_at(frequency=0, color='switchedoff')
            time.sleep(2)
            # Reset the intersection as a last step to avoid early rest by april tag reading
            self.curr_intersection_type = IntersectionType.Unknown

            # Publish go message
            message.data = True
            self.pub_intersection_go.publish(message)
            rospy.loginfo(f"[{self.node_name}] -> Go")


            # TODO Debug/Demo testing:
            if self.params["~tl_demo_mode"] or self.params["~ss_demo_mode"]:
                new_info = TagInfo()
                new_info.tag_type = new_info.SIGN
                new_tag_data = AprilTagsWithInfos()
                new_tag_data.infos.append(new_info)
                self.intersection_type_callback(new_tag_data) # Unknown
                # simulate stop sign
                if self.params["~ss_demo_mode"]:
                    new_info.traffic_sign_type = new_info.STOP
                    new_tag_data.infos.append(new_info)
                    self.intersection_type_callback(new_tag_data)
                # simulate TL
                if self.params["~tl_demo_mode"]:
                    new_info.traffic_sign_type = new_info.T_LIGHT_AHEAD
                    new_tag_data.infos.append(new_info)
                    self.intersection_type_callback(new_tag_data)

        elif action == ActionState.TimedOut:
            # Set color to red
            self.blink_at(frequency=0, color='red')
            time.sleep(2)
            # Publish timed-out message
            message.data = True
            self.pub_timed_out.publish(message)
            rospy.loginfo(f"[{self.node_name}] -> Timed-out")

    def run(self):
        rate = rospy.Rate(0.5)  # 1Hz
        while not rospy.is_shutdown():
            BaseComNode.run(self)
            rate.sleep()

            # TODO Needed to hold the bot still. Does not work and make the node hang, keep commented for now
            #car_cmd_msg = Twist2DStamped(v=0.0, omega=0.0)
            #car_cmd_msg.header.stamp = current_time_stamp
            #self.pub_coord_cmd.publish(car_cmd_msg)


if __name__ == '__main__':
    # create the node
    node = CommunicationNode(node_name='communication_node')
    node.run()
    # keep spinning
    rospy.spin()
