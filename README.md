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

## Check Installation

Run `./run_ci_checks.sh`. It should complete with all green successes in 5-10 seconds.
