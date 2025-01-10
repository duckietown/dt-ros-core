#!/usr/bin/env python3

import sys
import math
import numpy as np
import duckietown_code_utils as dtu

import rospy


from nav_msgs.msg import Odometry
from geometry_msgs.msg import Quaternion, Twist, Pose, Point, Vector3, TransformStamped, Transform, Polygon, Point32

from duckietown.dtros import DTROS, NodeType
from duckietown_msgs.msg import WheelEncoderStamped

from tf2_ros import TransformBroadcaster

from tf import transformations as tr
from duckietown.dtros import DTROS, NodeType, TopicType, DTParam, ParamType
from duckietown_msgs.msg import (
    Twist2DStamped,
    LanePose,
    WheelsCmdStamped,
    BoolStamped,
    FSMState,
    StopLineReading,
    ObstacleProjectedDetectionList
)



class Avoider(DTROS): #comment here 

    def __init__(self,node_name):
        ''' put all the init stuff like states, lane coef , planned path status vairable here '''
        self.stuff = 0

        super(Avoider, self).__init__(node_name=node_name, node_type=NodeType.PERCEPTION)

        #rospy.init_node("detector")
        # declare variables here

        self.state = 0 #-1 failed, 0 in work , 1 finished
        self.rate = rospy.Rate(0.2)

        self.OBSTACLE_RADIUS = 5 # in centimeters
        self.LANE_WIDTH = 25.5

        self.OBSTACLE_RADIUS = 5 # in centimeters
        self.LANE_WIDTH = 25.5

        self.pose_d = 0
        self.pose_phi = 0
        self.in_lane = False
        self.bool = False

        self.obstacle = [0,25] # obstacle in robo frame , units:cm

        
        self.pub_int_done = rospy.Publisher("~object_detected", BoolStamped, queue_size=1)

        self.sub_avoid_done = rospy.Subscriber("~avoidance_done", BoolStamped, self.reset_state)
        
        self.sub_lane_reading = rospy.Subscriber(
            "~lane_pose", LanePose, self.get_pose , queue_size=1 #TODO need to fix hardcoded values
        )

        self.object_reading = rospy.Subscriber(
            "~detections", ObstacleProjectedDetectionList, self.get_object , queue_size=1 #TODO need to fix hardcoded values
        )
        # subscribe to objects
        #TODO

        self.pub_path = rospy.Publisher(
            "~avoidance_path", Polygon, queue_size=1, dt_topic_type=TopicType.CONTROL
        )

        #self.rate.sleep()
        msg = ObstacleProjectedDetectionList()
        #self.get_object(msg)

    def reset_state(self,msg):
        if msg.data == True:
            self.bool = False


    def get_pose(self, input_pose_msg):

        self.pose_d =   input_pose_msg.d
        self.pose_phi = input_pose_msg.phi
        self.in_lane =  input_pose_msg.in_lane

        #print("got lane ")
        #print("Lane stuff ")
        #print(self.pose_d)
        #print(self.pose_phi)

    def get_object(self, msg):


        y = msg.detections[0].location.x * 100
        x = (msg.detections[0].location.y * 100) + 13
        self.obstacle = [x,y]
        # while(1):
        print(self.obstacle)
        if self.get_to_avoid(self.obstacle,[self.pose_d * (-100),self.pose_phi]):
                print(" making path ")
                self.plan(self.obstacle,[self.pose_d * (-100),self.pose_phi])




    def rotate(self,origin, point, angle):
        """
        Rotate a point counterclockwise by a given angle around a given origin.
    
        The angle should be given in radians.
        """
        ox, oy = origin
        px, py = point
    
        qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
        qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
        return qx, qy

    
    def plan(self,obstacle_info: np.array, lane_info: np.array):
        """
        Inputs: 
            obstacle_info contains the x and y position of the obstacle in robot frame(assume constant diameter?) 
                [x, y]
            lane_info contains distance from yellow lane (center) and orientation w.r.t lane 
                [d, theta]
        Outputs:
            waypoints: list of waypoints [(x_1,  y_1), (x_2, y_2), ..., (x_n, y_n)] in original robot frame
        """
        lane_info[0] = lane_info[0] + (self.LANE_WIDTH/2)
        
        obstacle_info = self.rotate((0,0), np.array(list(obstacle_info)), lane_info[1])

        #obstacle_info = np.array([30,40]) #TODO remove this 
    
        # Assume obstacle_info[1] > 20
        safe_distance = 15
        end_point = (0,  obstacle_info[1] + 2.2*safe_distance) # want to end up 20 centimeters past the obstacle, back at the same x coordinate
    
        # middle point
        if obstacle_info[0] < 0:
            if safe_distance + (lane_info[0] + obstacle_info[0]) < 20:
                middle_point = (max(obstacle_info[0] + safe_distance, end_point[0]), obstacle_info[1])
            else:
                middle_point = (min(obstacle_info[0] - safe_distance,end_point[0]), obstacle_info[1])
        else:
            middle_point = (min(obstacle_info[0] - safe_distance,end_point[0]), obstacle_info[1])
    
    
        first_third = (middle_point[0], middle_point[1]/2)
        second_third = (middle_point[0], middle_point[1] + (end_point[1] - middle_point[1])/2)
        
        waypoints = np.array([first_third, second_third, end_point])
        waypoints = [self.rotate((0,0), w, -lane_info[1]) for w in waypoints]

        #TODO 
        #insted of returning publish message directly
        # convert the axis as well as the units
        poly = Polygon()
        for p in waypoints:
            point = Point32()
            point.x = (p[1]/100)
            point.y = -(p[0]/100)
            point.z = 0

            poly.points.append(point)

        print("publishing path")
        
        self.pub_path.publish(poly)
        if self.bool == False:
            msg_done = BoolStamped()
            msg_done.data = True
            self.pub_int_done.publish(msg_done)
            self.bool = True
        self.rate.sleep()
            


        
        return 
    
    def get_to_avoid(self,obstacle_info: np.array, lane_info: np.array):
        """
        Inputs:
            same as plan
        Outpus:
            to_avoid: boolean of whether we need to avoid obstacle
        """
        
        lane_info[0] = lane_info[0] + (self.LANE_WIDTH/2)
        obstacle_info = self.rotate((0,0), np.array(list(obstacle_info)), lane_info[1])
        #obstacle_info = np.array([30,40]) #TODO remove this 
        #print("updated obstacle ",obstacle_info)    
        obstacle_2_lane_x = obstacle_info[0] + lane_info[0]
        to_avoid = np.logical_not(np.logical_or(obstacle_2_lane_x < -self.OBSTACLE_RADIUS, 
                             obstacle_2_lane_x > (self.LANE_WIDTH + self.OBSTACLE_RADIUS)))
        
        #print(to_avoid)
        return to_avoid
    



if __name__ == '__main__':
    A = Avoider(node_name="detect_obstacle")
    rospy.spin()
