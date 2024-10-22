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
   - zero the arm torque offsets:
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
   - Alias `launch_app` on compute system
   - Otherwise, run the following commands from the root of your ROS workspace:
        - `conda activate feed`
        - `source devel/setup.bash`
        - `cd ~/deployment_ws/src/feedingpage/vue-ros-demo`
        - `npm run serve` 
6. Run the feeding demo:
   - Alias `run_demo` on compute system
   - Otherwise,run the following commands from the root of your ROS workspace:
        - `conda activate feed`
        - `source devel/setup.bash`
        - `cd src/feeding-deployment/src/feeding_deployment/integration`
        - `python demo.py --run_on_robot`

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
- IP for webapp:" http://192.168.1.2:8080/#/home

## Check Installation

Run `./run_ci_checks.sh`. It should complete with all green successes in 5-10 seconds.
