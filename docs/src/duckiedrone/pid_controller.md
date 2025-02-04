# PID controller

The PID controller on the Duckiedrone takes a velocity or position setpoint and produces `SET_ATTITUDE_TARGET` mavlink commands that are sent through mavros to the flight controller.

It takes as input a state estimate form the StateEstimator node, containing an estimated pose and twist.

The `SET_ATTITUDE_TARGET` command is detailed [here](https://mavlink.io/en/messages/common.html#SET_ATTITUDE_TARGET). On the Ardupilot Copter firmware used on the flight controller the `thrust` field can be interpreted either as a commanded normalized thrust level or a climb rate. On the DD24 this is configured by default to be the normalized thrust.

The message is published to

`~setpoint_raw/attitude`

and remapped through a launch file to the mavros node topic

`/[ROBOT_NAME]/mavros/setpoint_raw/attitude`.