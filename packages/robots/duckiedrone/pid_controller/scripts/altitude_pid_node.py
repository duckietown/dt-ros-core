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

print('Connecting to vehicle on: %s' % connection_string)
vehicle = connect(connection_string)

print("Waiting for calibration...")
vehicle.send_calibrate_accelerometer(simple=True)
vehicle.send_calibrate_gyro()
print("Calibration completed")

pid_controller = PID(0.10, 0.05, 0.04, setpoint=target_altitude, sample_time=1/PID_SAMPLE_RATE, output_limits=(0, 1))

def range_callback(msg : Range):
    global error_data

    current_range = msg.range
    altitude = current_range
    error = pid_controller.setpoint - altitude
    error_data.append(error)

    thrust = pid_controller(altitude)
    print(f"Thrust [0-1]: {thrust}\n Error [m]: {error}\n")
    # set_attitude(thrust=0)
    set_attitude(thrust=thrust)

def altitude_controller():

    print("Arming motors")
    vehicle.mode = VehicleMode("GUIDED_NOGPS")
    
    set_attitude(thrust=0)
    vehicle.armed = True

    while not vehicle.armed:
        print(" Waiting for arming...")
        vehicle.armed = True
        time.sleep(1)

    print("Taking off!")

    rospy.Subscriber(f"/{ROBOT_NAME}/bottom_tof_driver_node/range", Range, range_callback, queue_size=1)
    rospy.spin()

def send_attitude_target(roll_angle=0.0, pitch_angle=0.0, yaw_angle=None, yaw_rate=0.0, use_yaw_rate=False, thrust=0.5):
    if yaw_angle is None:
        yaw_angle = vehicle.attitude.yaw

    msg = vehicle.message_factory.set_attitude_target_encode(
    0, 1, 1, 0b00000000 if use_yaw_rate else 0b00000100,
    to_quaternion(roll_angle, pitch_angle, yaw_angle),
    0, 0, math.radians(yaw_rate), thrust
    )
    vehicle.send_mavlink(msg)

def set_attitude(roll_angle=0.0, pitch_angle=0.0, yaw_angle=None, yaw_rate=0.0, use_yaw_rate=False, thrust=0.5, duration=0):
    send_attitude_target(roll_angle, pitch_angle, yaw_angle, yaw_rate, use_yaw_rate, thrust)
    start = time.time()
    while time.time() - start < duration:
        send_attitude_target(roll_angle, pitch_angle, yaw_angle, yaw_rate, use_yaw_rate, thrust)
    # time.sleep(0.1)
    send_attitude_target(0, 0, 0, 0, True, thrust)

def to_quaternion(roll=0.0, pitch=0.0, yaw=0.0):
    t0 = math.cos(math.radians(yaw * 0.5))
    t1 = math.sin(math.radians(yaw * 0.5))
    t2 = math.cos(math.radians(roll * 0.5))
    t3 = math.sin(math.radians(roll * 0.5))
    t4 = math.cos(math.radians(pitch * 0.5))
    t5 = math.sin(math.radians(pitch * 0.5))

    w = t0 * t2 * t4 + t1 * t3 * t5
    x = t0 * t3 * t4 - t1 * t2 * t5
    y = t0 * t2 * t5 + t1 * t3 * t4
    z = t1 * t2 * t4 - t0 * t3 * t5

    return [w, x, y, z]

if __name__ == '__main__':
    rospy.init_node('altitude_pid_node', anonymous=True)
    try:
        altitude_controller()
    except rospy.ROSInterruptException:
        pass
    finally:
        print("Setting LAND mode...")
        vehicle.mode = VehicleMode("LAND")
        time.sleep(1)
        print("Closing vehicle object")
        vehicle.close()
        if sitl is not None:
            sitl.stop()
        print("Completed")
