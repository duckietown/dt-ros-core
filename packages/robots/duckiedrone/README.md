# dt-drone-dev-notes

(This was copied from the [dt-drone-dev-notes](https://github.com/duckietown/dt-drone-dev-notes) repo, to be updated)
How to run stuff, which repos are related ...

### Basic info
* Default branch to use for the repositories: `ente`

## Repositories (details in sections below)
* [dt-duckiebot-interface](https://github.com/duckietown/dt-duckiebot-interface): Drivers
* [dt-drone-interface](https://github.com/duckietown/dt-drone-interface): Collection of drone packages (altitude, state est, pid, joystick mapper (old))
* [dt-drone-commands-mux](https://github.com/duckietown/dt-drone-commands-mux): Fly commands switch, manual vs. autonomous
* [dt-drone-web-joystick-js](https://github.com/duckietown/dt-drone-web-joystick-js): Web joystick (publish to manual input of the above repo)
* [dt-drone-vision](https://github.com/duckietown/dt-drone-vision): Optical flow

## Workflow

* power on the Pi, check can `ping` and can `docker -H [hostname].local ps`

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
        * Arming and disarming: in the same window as IMU Calibration, select the Service: `/[hostname]/flight_controller_node/set_mode`. **Note: this will make the motors start spinning. Make sure the protections are equipped, or the propellers are not mounted!** To **arm**, in the `mode - uint8 - 0` row, double-click the value `0`, and change it to `1`, then click the `Call` button. The motors should start spinning. To **disarm**, set the value to `0` and click the `Call` button.
  * To further make the drone accept fly commands from other nodes, in the `set_mode` service above, the mode should be set from `0` -> `1` -> `2`.

### dt-drone-interface
* Actions:
  * Clone repo to local machine
  * (*Optional*) `dts devel build -f -H [hostname]` (same as above repo)
  * To start altitude node:
    * `dts devel run -f -s -M --name drone-iface-alt -H [hostname] -c /bin/bash`, and
    * `./launchers/altitude-node.sh`
    * wait until the log appears: `/altitude_node] Health status changed [STARTING] -> [STARTED]`
  * To start the altitude controller experiment (from Vincenzo)
    * In another terminal, same repo: `dts devel run -f -s -M --name drone-iface-alt-ctrl -H [hostname] -c /bin/bash`, and
    * `./launchers/altitude-control.sh`
* Expected outcome:
  * Verify altitude data in `rqt`: `Plugins->Visualization->Plot`. Set Topic: `/[hostanme]/altitude_node/altitude/range` and click the `+` sign. Try moving up and tilt.
  * Check the controller values in the terminal for `drone-iface-alt-ctrl`, the Error and PID info should look like: `Error: 0.2970230087637901, P: 47.52368140220642, I: 298.78502361307255, D: 0.9097172658959242`
    * The output commands of the altitude controller node is at `/[hostname]/fly_commands_mux_node/autonomous`. This will not be used by the FC unless
      1. the drone mode is set from `0` -> `1` -> `2`
      2. the `fly_commands_mux_node` is started (coming up next)

### dt-drone-commands-mux
* Actions:
  * Clone repo to local machine
  * (*Optional*) Build: `dts devel build -f -H [hostname]`
  * Run: `dts devel run -f -s -M -H [hostname]`
    * Wait until the log appears: `/fly_commands_mux_node] Initialization completed.`
* Expected outcome:
  * `dts start_gui_tools --name cmd_mux [hostname]`
    * `rosnode info /[hostname]/fly_commands_mux_node`
    * verify in the Publications: `/[hostname]/flight_controller_node/commands`
    * verify in the Subscriptions:
      * `/[hostname]/fly_commands_mux_node/manual`
      * `/[hostname]/fly_commands_mux_node/autonomous`

### dt-drone-web-joystick-js
* Actions:
  * Clone repo to local machine
  * Open the folder and double-click the `index.html` file to open in a browser (tested in Chrome)
  * Check the repo readme for usage
* Outcome:
  * Control the drone manually

### dt-drone-vision
Check repo README for how to build, run and verify results.

### Minimal altitude PID
* Actions:
  * From a `ente` shell update the Duckiedrone with `dts duckiebot update [ROBOT_NAME]`
  * Clone repo to local machine
  * (*Optional*) Build: `dts devel build`
  * Run: `dts devel run -R [ROBOT_NAME] -c "python3 packages/robots/duckiedrone/pid_controller/scripts/altitude_pid_node.py"`

* Expected outcome:
  * **[DANGER]**: The drone will drift on the x-y (horizontal) plane, make sure to have enough space around it.
  * **[INFO]**: If you want to land the drone press `CTRL`+`C` to stop the script.
  * Drone should arm and takeoff to a height of 1 m.
  * In the console you should see a message reporting the thrust value and the range error being printed continuously:
    
    ```
    Thrust [0-1]: 0.43513294704555977
    Error [m]: 0.013999998569488525
    ```
  * At steady state (Error=0 m) the thrust value should be about 0.4