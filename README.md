## Requirements

- Python 3.10+
- Tested on Ubuntu 20.04

## Pre-Installation

1. Install ROS and rospy.
2. Install [pyaudio](https://pypi.org/project/PyAudio/).

## Installation

1. Recommended: create and source a virtualenv.
2. `pip install -e ".[develop]"`

## Controller Example

1. On one terminal on the compute system, ssh to the NUC: `sshnuc` with lab password
2. On the NUC, run the controller server: `cd feeding-deployment/src/feeding_deployment/robot_controller && python arm_server.py`
3. On another terminal on the compute system, run the controller client example: `cd deployment_ws/src/feeding-deployment/src/feeding_deployment/robot_controller && python arm_client.py`

## Run Feeding Demo on Real Robot
1. Run the arm controller server on the NUC:
   - ssh to the NUC: `sshnuc` with lab password
   - run the controller server:
        - Alias `run_server` on NUC
        - Otherwise, run the following commands:
             - `conda activate controller`
             - `cd feeding-deployment/src/feeding_deployment/robot_controller`
             - `python arm_server.py`
2. Launch the roslaunch on compute system for visualization / publish tfs:
   - Alias `launch_robot` on compute system
   - Otherwise,run the following commands from the root of your ROS workspace:
        - `conda activate feed`
        - `source devel/setup.bash`
        - `cd src/feeding-deployment/launch`
        - `roslaunch robot.launch`
3. Run the feeding demo:
   - Alias `run_demo` on compute system
   - Otherwise,run the following commands from the root of your ROS workspace:
        - `conda activate feed`
        - `source devel/setup.bash`
        - `cd src/feeding-deployment/src/feeding_deployment/integration`
        - `python demo.py --run_on_robot`
  
## Run Feeding Demo in Simulation
1. Launch the roslaunch for visualization / publish tfs:
   - Navigate to the launch files: `cd launch`
   - Launch: `roslaunch sim.launch`
2. Run the feeding demo:
   - Navigate to integration scripts: `cd src/feeding_deployment/integration`
   - Run demo: `python demo.py`

## Check Installation

Run `./run_ci_checks.sh`. It should complete with all green successes in 5-10 seconds.
