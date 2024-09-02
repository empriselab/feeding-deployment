"""This is a test to make sure that collisions between a held drink and the conservative bounding box are caught."""


from pybullet_helpers.link import get_relative_link_pose
from pybullet_helpers.inverse_kinematics import check_collisions_with_held_object, set_robot_joints_with_held_object
from feeding_deployment.simulation.scene_description import (
    SceneDescription,
)
from feeding_deployment.simulation.simulator import (
    FeedingDeploymentPyBulletSimulator,
)
import pybullet as p


def _main() -> None:

    scene_description = SceneDescription()
    sim = FeedingDeploymentPyBulletSimulator(scene_description)

    # Set up the simulator so the drink is held.
    sim.held_object_name = "drink"
    sim.held_object_id = sim.drink_id
    sim.robot.set_finger_state(scene_description.tool_grasp_fingers_value)
    finger_frame_id = sim.robot.link_from_name("finger_tip")
    end_effector_link_id = sim.robot.link_from_name(sim.robot.tool_link_name)
    drink_from_end_effector = get_relative_link_pose(
        sim.robot.robot_id, finger_frame_id, end_effector_link_id, sim.physics_client_id
    )
    sim.held_object_tf = drink_from_end_effector

    # Move the robot to a position where a collision is expected.
    joints = [2.8884768101246143, -0.7913320348241513, -1.7742571378056136, -2.078073911389284, 2.2868481461996795, -0.8264030187967055, -0.11233229012519357, 0.44, 0.44, 0.44, 0.44, -0.44, -0.44]
    
    while True:
        set_robot_joints_with_held_object(sim.robot, sim.physics_client_id,
                                        sim.held_object_id, sim.held_object_tf,
                                        joints)
        collision = check_collisions_with_held_object(sim.robot, sim.get_collision_ids(), sim.physics_client_id, sim.held_object_id,
                                        sim.held_object_tf, sim.robot.get_joint_positions())
        print("Collision detected?", collision)
        p.stepSimulation(physicsClientId=sim.physics_client_id)
        import time; time.sleep(0.01)


if __name__ == "__main__":
    _main()
