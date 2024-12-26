# import urdf "robot.urdf" to pybullet world

import pybullet as p
import pybullet_data
import os

# Set up the PyBullet environment
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())

# Load the robot
robot_urdf = os.path.join(os.path.dirname(__file__), "robot.urdf")
print("Robot URDF path:", robot_urdf)
robot_id = p.loadURDF(robot_urdf)

# Run the simulation
while True:
    p.stepSimulation()


