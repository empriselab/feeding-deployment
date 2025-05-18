## Requirements

- Python 3.10+
- Tested on Ubuntu 20.04

## Pre-Installation

1. Install ROS and rospy.
2. Install [pyaudio](https://pypi.org/project/PyAudio/).

## Installation

1. Recommended: create and source a virtualenv.
2. `pip install -e ".[develop]"`

## Run Feeding Demo on Real Robot
1. Run the arm controller server on the NUC:
   - ssh to the NUC: `sshnuc` with lab password
   - [only for inside-mouth bite transfer] zero the arm torque offsets:
        - Alias `set_zeros` on NUC
        - Otherwise, run the following commands:
             - `conda activate controller`
             - `cd ~/feeding-deployment/src/feeding_deployment/robot_controller`
             - `python kinova.py`
   - run the controller server:
        - Alias `run_server` on NUC
        - Otherwise, run the following commands:
             - `conda activate controller`
             - `cd feeding-deployment/src/feeding_deployment/robot_controller`
             - `python arm_server.py`
2. Run bulldog on the NUC:
   - ssh to the NUC: `sshnuc` with lab password
   - run bulldog with alias `run_bulldog`
2. Run a roscore on the compute system: `roscore`
3. Launch all the sensors on the compute system using `launch_sensors`
3. Launch the roslaunch on compute system for visualization / publish tfs:
   - Alias `launch_robot` on compute system
   - Otherwise,run the following commands from the root of your ROS workspace:
        - `conda activate feed`
        - `source devel/setup.bash`
        - `cd src/feeding-deployment/launch`
        - `roslaunch robot.launch`
4. Start feeding utensil:
   - Alias `launch_utensil` on compute system
   - Otherwise, run the following commands from the root of your ROS workspace:
        - `conda activate feed`
        - `source devel/setup.bash`
        - `rosrun wrist_driver_ros wrist_driver`  
   - _Important Note:_ To shutdown this node, press Ctrl + / (Signal handling is setup to shutdown cleanly)
5. Start the web application:
   - Make sure that the feeding laptop's WiFi is off (so that the webapp only launches on the router IP)
   - Alias `launch_app` on compute system
   - Otherwise, run the following commands from the root of your ROS workspace:
        - `conda activate feed`
        - `source devel/setup.bash`
        - `cd ~/deployment_ws/src/feedingpage/vue-ros-demo`
        - `npm run serve`
   - On a browser connected to FeedingDeployment-5G (on the laptop or the iPad), open the following webpage: `http://192.168.1.2:8080/#/task_selection`  
6. Run the feeding demo:
   - Make sure that the feeding laptop's WiFi is on and connected to the internet so that ChatGPT API works (use KortexWiFi if available)
   - Alias `run_demo` on compute system
   - Otherwise,run the following commands from the root of your ROS workspace:
        - `conda activate feed`
        - `source devel/setup.bash`
        - `cd src/feeding-deployment/src/feeding_deployment/integration`
        - `python run.py --user tests --run_on_robot --use_interface --no_waits`
   - _Important Note 1:_ If you want to resume from some state (state names: after_utensil_pickup, after_bite_pickup, last_state), use: `python run.py --user tests --run_on_robot --use_interface --no_waits --resume_from_state after_utensil_pickup` (replace after_utensil_pickup with appropriate state name).
   - _Important Note 2:_ The preset food item for `tests` user is bananas. If you want to try some other food item, just change the user name to a new one. For example, `python run.py --user tests_new --run_on_robot --use_interface --no_waits`

### Moving the robot to preset configurations

You can move the robot to preset configurations by running:
- Alias `cd_actions` on compute system
- `python retract.py` (you can also send it to transfer.py and acquisition.py) 

### Calibrate tool offset for inside-mouth transfer

1. Grasp the tool and move to before bite transfer position.
2. Calibrate tool:
   - Alias `cd_demo` on compute system
   - Otherwise, run the following commands from the root of your ROS workspace:
        - `conda activate feed`
        - `source devel/setup.bash`
        - `cd src/feeding-deployment/src/feeding_deployment/integration`
   - `python transfer_calibration.py --tool <tool_name>` where <tool_name> is one of "fork", "drink" and "wipe"
3. Manually (using buttons on the robot) move the robot to the intended inside-mouth transfer config, and press [ENTER] in the script above to record it. 
4. To test the tool calibration:
   - Alias `cd_demo` on compute system
   - Otherwise, run the following commands from the root of your ROS workspace:
        - `conda activate feed`
        - `source devel/setup.bash`
        - `cd src/feeding-deployment/src/feeding_deployment/integration` 
   - `python transfer_calibration.py --tool <tool_name> --test` where <tool_name> is one of "fork", "drink" and "wipe"
  
## Run Feeding Demo in Simulation
1. Launch the roslaunch for visualization / publish tfs:
   - Navigate to the launch files: `cd launch`
   - Launch: `roslaunch sim.launch`
2. Run the feeding demo:
   - Navigate to integration scripts: `cd src/feeding_deployment/integration`
   - Run demo: `python demo.py`

## Random

- To check FT readings: `rostopic echo /forque/forqueSensor`
- IP for robot: 192.168..10
- IP for webapp: `http://192.168.1.2:8080/#/task_selection`

## Check Installation

Run `./run_ci_checks.sh`. It should complete with all green successes in 5-10 seconds.
