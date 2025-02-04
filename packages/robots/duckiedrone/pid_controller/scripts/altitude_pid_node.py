#!/usr/bin/env python

"""
altitude_pid_node.py: (Copter Only)

This example shows a PID controller regulating the altitude of Copter using a downward-facing rangefinder sensor.

Caution: A lot of unexpected behaviors may occur in GUIDED_NOGPS mode.
    Always watch the drone movement, and make sure that you are in a dangerless environment.
    Land the drone as soon as possible when it shows any unexpected behavior.

Tested in Python 3.12.3

"""
import collections
import collections.abc
import rospy
from sensor_msgs.msg import Range
from simple_pid import PID
from dronekit import connect, VehicleMode
from pymavlink import mavutil # Needed for command message definitions
import time
import math
import matplotlib.pyplot as plt
import os
from mavros_msgs.msg import AttitudeTarget
from mavros_msgs.srv import CommandBool, SetMode

# The collections module has been reorganized in Python 3.12 and the abstract base
# classes have been moved to the collections.abc module. This line is necessary to
# fix a bug in importing the MutableMapping class in `dronekit`.
collections.MutableMapping = collections.abc.MutableMapping

PID_SAMPLE_RATE = 50 # [Hz]
ROBOT_NAME = os.environ.get('VEHICLE_NAME', None)
target_altitude = 1
range_zeroing_offset = 0.0
error_data = []

# Connect to the Vehicle
connection_string = rospy.get_param('~connection_string', None)
sitl = None

connection_string = F"tcp:{ROBOT_NAME}.local:5760"

# print('Connecting to vehicle on: %s' % connection_string)
# vehicle = connect(connection_string)

# print("Waiting for calibration...")
# vehicle.send_calibrate_accelerometer(simple=True)
# vehicle.send_calibrate_gyro()
# print("Calibration completed")

pid_controller = PID(0.10, 0.05, 0.04, setpoint=target_altitude, sample_time=1/PID_SAMPLE_RATE, output_limits=(0, 1))

def range_callback(msg : Range):
    global error_data

    current_range = msg.range
    altitude = current_range
    error = pid_controller.setpoint - altitude
    error_data.append(error)

    thrust = pid_controller(altitude)
    print(f"Thrust [0-1]: {thrust}\n Error [m]: {error}\n")
    
    msg = AttitudeTarget()
    
    msg.thrust = thrust
    # msg.orientation.w = 1
    
    pub_setpoint.publish(msg)

def altitude_controller():
    global pub_setpoint

    print("Arming motors")
    # Wait for the service to become available
    arm_vehicle()
    print("Vehicle armed")
    
    set_mode_guided_no_gps()
    # vehicle.mode = VehicleMode("GUIDED_NOGPS")
    
    # vehicle.armed = True

    # while not vehicle.armed:
        # print(" Waiting for arming...")
        # vehicle.armed = True
        # time.sleep(1)

    # print("Taking off!")

    pub_setpoint = rospy.Publisher(f"/{ROBOT_NAME}/mavros/setpoint_raw/attitude", AttitudeTarget)
    rospy.Subscriber(f"/{ROBOT_NAME}/bottom_tof_driver_node/range", Range, range_callback, queue_size=1)
    rospy.spin()

def set_mode_guided_no_gps():
    rospy.wait_for_service(f"/{ROBOT_NAME}/flight_controller_node/set_mode")

    # Create a service proxy
    set_mode_service = rospy.ServiceProxy(f"/{ROBOT_NAME}/flight_controller_node/set_mode", SetMode)
    
    # Call the service
    response = set_mode_service(custom_mode="GUIDED_NOGPS")
    
    # Check the response
    if response.mode_sent:
        rospy.loginfo("Successfully called the set mode service")
    else:
        rospy.logwarn("Failed to call the set mode service")

def arm_vehicle():
    rospy.wait_for_service(f"/{ROBOT_NAME}/flight_controller_node/arm")

    # Create a service proxy
    arm_service = rospy.ServiceProxy(f"/{ROBOT_NAME}/flight_controller_node/arm", CommandBool)
    
    # Call the service
    response = arm_service(True)
    
    # Check the response
    if response.success:
        rospy.loginfo("Successfully called the arm service")
    else:
        rospy.logwarn("Failed to call the arm service")

if __name__ == '__main__':
    rospy.init_node('altitude_pid_node', anonymous=True)
    try:
        altitude_controller()
    except rospy.ROSInterruptException:
        pass
    finally:
        time.sleep(1)
        print("Closing vehicle object")
        if sitl is not None:
            sitl.stop()
        print("Completed")
