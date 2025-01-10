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