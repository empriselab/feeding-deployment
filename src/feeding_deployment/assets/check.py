# import urdf "robot.urdf" to pybullet world

import pybullet as p
import pybullet_data
import os

# Set up the PyBullet environment
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())

# Load the robot
# robot_urdf = os.path.join(os.path.dirname(__file__), "robot/robot.urdf")
# print("Robot URDF path:", robot_urdf)
# robot_id = p.loadURDF(robot_urdf)

# base_urdf = os.path.join(os.path.dirname(__file__), "vention_base/vention_base.urdf")
# print("Base URDF path:", base_urdf)
# base_id = p.loadURDF(base_urdf, basePosition=[0, 0, -0.01])

# microwave_urdf = os.path.join(os.path.dirname(__file__), "microwave/mobility.urdf")
# print("Microwave URDF path:", microwave_urdf)
# microwave_id = p.loadURDF(microwave_urdf, basePosition=[0, 0, 0], useFixedBase=True)

# refridgerator_urdf = os.path.join(os.path.dirname(__file__), "refridgerator/mobility.urdf")
# print("Refridgerator URDF path:", refridgerator_urdf)
# refridgerator_id = p.loadURDF(refridgerator_urdf, basePosition=[0, 0, 0], useFixedBase=True)

sink_urdf = os.path.join(os.path.dirname(__file__), "sink/sink.urdf")
print("Sink URDF path:", sink_urdf)
sink_id = p.loadURDF(sink_urdf, basePosition=[0, 0, 0], useFixedBase=True)

# Run the simulation
input("Press Enter to start the simulation...")
while True:
    p.stepSimulation()
