
# Communication node

In its **default configuration** the communication_node is intended to run with the full integrated setup (Lane following and FSM handled continues running).  

It can also be tested/demoed in a standalone manner. It can be run in an indefinite setup to demonstrate or debug its functionalities (Traffic Light or Stop Sign).  

In order to test either the Traffic Light or the Stop Sign solving, follow these instructions to modify the configuration and use the appropriate instructions to launch in either mode.  

**IMPORTANT**:
To run the code make sure you have an up to date dts shell and duckiebot.  
You can run the following commands:  
`dts update`  
`dts desktop update`  
`dts duckiebot update YOURBOTNAME`  
And for an accurate LED control and consistent LED flashing, always execute the following command after the bot has been reboot:  
`ssh duckie@YOURBOTNAME.local "echo '400000' | sudo tee /sys/bus/i2c/devices/i2c-1/bus_clk_rate"`  


## 1) Traffic Light standalone debug/demo mode  

In the `communication_node` config yaml file (under `mooc-exercises/project/solution/src/dt-core/packages/communication/config/communication_node/default.yaml`) change the `tl_demo_mode` to `True`. This will emulate a TL intersection type and make the code run indefinitely in this mode (Make sure that `ss_demo_mode` is set to `False`).  

Then you need to build it (from `mooc-exercises/project` folder) using the command:  
`dts exercises build`  

Place the bots (2 or more) at a Traffic Light intersection and run the code for all the bots. 

Run the code from the appropriate `mooc-exercises/project` folder. You need to pull and checkout the correct branch (For example `korra141/mooc-exercises/daffy-project`)  
`dts exercises test --duckiebot_name YOURBOTNAME`  
Note: For the sake of completeness of information, inside the `run.sh` in the `mooc-exercises/project` the communication_node is launched by this specific command `dt-exec-FG roslaunch --wait duckietown_demos communication.launch || true`  

The bot will start reading the TL signal and the code will run indefinitely. When it reads the GREEN light it will set its LEDs to GREEN. There is a possibility that the bot reaches a time-out state if it never reads GREEN for a certain time period (defined in config yaml file as `time_out_sec`). The LEDs will turn RED to indicate the time-out state.  


## 2) Stop Sign standalone debug/demo mode  

In the `communication_node` config yaml file (under `mooc-exercises/project/solution/src/dt-core/packages/communication/config/communication_node/default.yaml`) change the `ss_demo_mode` to `True`. This will emulate a Stop Sign intersection type and make the code run indefinitely in this mode (Make sure that `tl_demo_mode` is set to `False`).  

**IMPORTANT**: Due to a limitation in the LED control related to the `fifo-bridge` update rate, the LED flashing can be very inconsistent which greatly affects the LED frequency reading of other bots. This can cause the Stop Sign solving to be very inconsistent and erroneous at times. A work around to this would be to launch the communication_node in the standalone demo mode using (b) `dts devel run` instead of (a) `dts exercises test`.


### a) Stop Sign standalone using usual dts exercises test  
You need to build it (from `mooc-exercises/project` folder) using the command:  
`dts exercises build`  

Place the bots (2 or more) at a Stop Sign intersection and run the code for all the bots. Make sure all bots can see each others LEDs.

Run the code from the appropriate `mooc-exercises/project` folder. You need to pull and checkout the correct branch (For example `korra141/mooc-exercises/daffy-project`)  
`dts exercises test --duckiebot_name YOURBOTNAME`  
Note: For the sake of completeness of information, inside the `run.sh` in the `mooc-exercises/project` the communication_node is launched by this specific command `dt-exec-FG roslaunch --wait duckietown_demos communication.launch || true`  

The bot will start flashing its LEDs and the negotiation will run indefinitely. Bots that get the priority will set their LEDs to GREEN. Bots that don't get priority in the current negotiation cycle will set their LEDs to BLUE. There is a possibility a bot reaches a time-out state if it does not get a priority for a certain period of time (defined in config yaml file as `time_out_sec`). The LEDs will turn RED to indicate the time-out state.  


### b) Stop Sign standalone using usual dts devel run (Best)  
Note: This config was tested with the code in `korra141/dt-core` repo and `daffy-project-standalone` branch. In case the merged code in the `duckietown/dt-core` repo does not work using `dts devel` command, you can try the former.   

**IMPORTANT** Using `dts devel` the code needs to be build and run from the `dt-core` root folder. You also need to modify the `dt-core/launchers/default.sh` file to launch the communication_node. Just replace the `dt-launcher-default-${ROBOT_TYPE}` line by `dt-exec roslaunch --wait duckietown_demos communication.launch`.  

Then you need to build it on the bot  
`dts devel build -f -H YOURBOTNAME.local`   

Finally, run the code (from the root `dt-core` folder)  
`dts devel run -H YOURBOTNAME.local`  

Similar to the above config (a), the bot will start flashing its LEDs and the negotiation will run indefinitely. Bots that get the priority will set their LEDs to GREEN. Bots that don't get priority in the current negotiation cycle will set their LEDs to BLUE. There is a possibility a bot reaches a time-out state if it does not get a priority for a certain period of time (defined in config yaml file as `time_out_sec`). The LEDs will turn RED to indicate the time-out state.  


# dt-core

This node contains the software for the core of the Duckietown stack, enabling autonomous driving on the Duckiebot and autonomous flight on the Duckiedrone.

## Workflow

* power on your robot, check can `ping` and can `docker -H [hostname].local ps`

---
* ***Note: the code is always cloned to the local computer, not the Pi on the robot***
* ***`[hostname]` is the hostname of the robot in the following context***
### dt-duckiebot-interface
* Actions:
    * Clone repo to local machine
    * To disable (some/all) heartbeat checks:
      * In file `packages/flight_controller_driver/config/flight_controller_node/default.yaml`, change the values in "heartbeats" section to `false`.
    * (*Optional, see below*) `dts devel build -f -H [hostname]`
      * in `ente` branch, if no error occur in the following command(s), the above step can be skipped
    * `dts devel run -f -s -M -H [hostname] -- --privileged`
* Expected outcome:
  * In the terminal the last above command is run, one should wait until the following message appears: `published the first image`
  * Verify the various drivers work properly with rqt:
    * `dts start_gui_tools --name db-iface [hostname]`
      * `rqt &`, then in the rqt window popped up, choose the plugin from the menu bar and select the topics
        * Check IMU reading: `Plugins->Visualization->Plot`. Set Topic: `/[hostname]/flight_controller_node/imu/linear_acceleration` and click the `+` sign
        * Check motors: `Plugins->Visualization->Plot`. Set Topic: `/[hostname]/flight_controller_node/motors/m1` (and `m2`, `m3`, `m4`) and click the `+` sign
        * ToF: `Plugins->Visualization->Plot`. Set Topic: `/[hostname]/bottom_tof_driver_node/range/range` (yes, range twice) and click the `+` sign.
        * Calibrate IMU: `Plugins->Services->Service Caller`. Select Service: `/[hostname]/flight_controller_node/calibrate_imu`. Click `Call` button and wait for the Response. The response should show `success - bool - True`
        * Arming and disarming: in the same window as IMU Calibration, select the Service: `/ing autonomous driving and 

### dt-drone-interface

See the readme in the [drone stack](packages/robots/duckiedrone/README.md)