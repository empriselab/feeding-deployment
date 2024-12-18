import numpy as np
from scipy.spatial.transform import Rotation as R

try:
    # ros imports
    import rospy
    import tf2_ros
    from geometry_msgs.msg import Pose, TransformStamped
    from sensor_msgs.msg import JointState
    from visualization_msgs.msg import Marker, MarkerArray

    from wrist_driver_interfaces.msg import SimpleJointAngleCommand
    from wrist_driver_interfaces.srv import SetWristMode, SetWristModeRequest, SetWristModeResponse
except ModuleNotFoundError as e:
    print(f"ROS not imported: {e}")
    ROSPY_IMPORTED = False

class HorizontalSpoon:
    def __init__(self):

        assert ROSPY_IMPORTED, "ROS is required to run Horizontal Spoon Controller"

        self.wrist_state_pub = rospy.Publisher('/cmd_wrist_joint_angles', SimpleJointAngleCommand, queue_size=10)

        self.last_pitch_angle = 0.0

        # subscribe to robot cartesian state
        self.cartesian_state_sub = rospy.Subscriber('/robot_cartesian_state', Pose, self.cartesian_state_callback)

        # set wrist control mode to velocity
        rospy.wait_for_service('set_wrist_mode')
        try:
            set_wrist_mode = rospy.ServiceProxy('set_wrist_mode', SetWristMode)
            resp1 = set_wrist_mode(0)
            if resp1.success:
                print("Successfully set wrist mode to velocity")
            else:
                print("Failed to set wrist mode to velocity")
        except rospy.ServiceException as e:
            print("Service call failed: %s"%e)

    def set_wrist_state(self, pitch, roll, vel=4):

        desired_pitch = -pitch
        desired_roll = -roll

        wrist_joint_states = rospy.wait_for_message('/wrist_joint_states', JointState)

        pitch_achieved = np.abs(wrist_joint_states.position[0] - desired_pitch) < 0.01
        roll_achieved = np.abs(wrist_joint_states.position[1] - desired_roll) < 0.01
        while (not pitch_achieved) or (not roll_achieved):
            wrist_joint_states = rospy.wait_for_message('/wrist_joint_states', JointState)
            current_pitch = wrist_joint_states.position[0]
            current_roll = wrist_joint_states.position[1]

            pitch_achieved = np.abs(current_pitch - desired_pitch) < 0.01
            roll_achieved = np.abs(current_roll - desired_roll) < 0.01
            
            wrist_state = SimpleJointAngleCommand()
            if not pitch_achieved:
                if np.abs(desired_pitch - current_pitch) > 0.2:
                    wrist_state.q0 = 0.8 * np.sign(desired_pitch - current_pitch)
                else:
                    #wrist_state.q0 = (desired_pitch - current_pitch) * vel
                    wrist_state.q0 = (desired_pitch - current_pitch) * 4
                    #wrist_state.q0 = (desired_pitch - current_pitch) * 8
            else:
                wrist_state.q0 = 0

            if not roll_achieved:
                if np.abs(desired_roll - current_roll) > 0.2:
                    wrist_state.q1 = 1.0 * np.sign(desired_roll - current_roll)
                else:
                    wrist_state.q1 = (desired_roll - current_roll) * vel
                    #wrist_state.q1 = (desired_roll - current_roll) * 8
            else:
                wrist_state.q1 = 0

            self.wrist_state_pub.publish(wrist_state)

            # print("q0: ", wrist_state.q0)
            # print("q1: ", wrist_state.q1)

        wrist_state = SimpleJointAngleCommand()
        wrist_state.q0 = 0
        wrist_state.q1 = 0
        self.wrist_state_pub.publish(wrist_state)

        # print("Wrist state set to: ", desired_pitch, desired_roll)
    
    def cartesian_state_callback(self, msg):
        self.cartesian_state = msg
        # print(self.cartesian_state)

        orientation = self.cartesian_state.orientation

        input_frame = np.eye(4)
        input_frame[:3, :3] = R.from_quat([orientation.x, orientation.y, orientation.z, orientation.w]).as_matrix()
        input_frame[:3, 3] = np.array([self.cartesian_state.position.x, self.cartesian_state.position.y, self.cartesian_state.position.z])

        # convert orientation to rotation matrix
        r = R.from_quat([orientation.x, orientation.y, orientation.z, orientation.w])
        r = r.as_matrix()

        x_axis = r[:, 0]
        y_axis = r[:, 1]
        z_axis = r[:, 2]
        z = np.array([0, 0, 1])

        cross = np.cross(x_axis, z) / np.linalg.norm(np.cross(x_axis, z))
        negative_cross = np.cross(z, x_axis) / np.linalg.norm(np.cross(z, x_axis))

        # visualize cross
        cross_frame = np.eye(4)
        cross_frame[:3, 0] = x_axis
        cross_frame[:3, 1] = np.cross(cross, x_axis) / np.linalg.norm(np.cross(cross, x_axis))
        cross_frame[:3, 2] = cross
        cross_frame[:3, 3] = np.array([self.cartesian_state.position.x, self.cartesian_state.position.y, self.cartesian_state.position.z])


        # visualize negative cross
        negative_cross_frame = np.eye(4)
        negative_cross_frame[:3, 0] = x_axis
        negative_cross_frame[:3, 1] = np.cross(negative_cross, x_axis) / np.linalg.norm(np.cross(negative_cross, x_axis))
        negative_cross_frame[:3, 2] = negative_cross
        negative_cross_frame[:3, 3] = np.array([self.cartesian_state.position.x, self.cartesian_state.position.y, self.cartesian_state.position.z])

        
        # np.set_printoptions(precision=2, suppress=True)
        # print(f"Cross: {cross} | Negative Cross: {negative_cross}")

        if np.linalg.norm(np.cross(z, x_axis)) > 0.2:
            if np.dot(cross, z_axis) > np.dot(negative_cross, z_axis):
            # if True:
                # print("Cross is the nearer vector")
                # cross is the nearer vector   
                direction = np.cross(z_axis, cross) / np.linalg.norm(np.cross(z_axis, cross))
                direction = np.dot(direction, x_axis)
                # print(" --- Pitch Direction: ", direction)
                angle = np.arccos(np.dot(z_axis, cross))
                pitch_angle = direction * angle
            else:
                # print("Negative Cross is the nearer vector")
                # negative_cross is the nearer vector
                direction = np.cross(z_axis, negative_cross) / np.linalg.norm(np.cross(z_axis, negative_cross))
                direction = np.dot(direction, x_axis)
                # print("Direction: ", direction)
                angle = np.arccos(np.dot(z_axis, negative_cross))
                pitch_angle = direction * angle
        else:
            pitch_angle = self.last_pitch_angle

        # print("Pitch angle: ", np.degrees(pitch_angle))

        # Find the angle between y-axis of cross

        roll_cross = np.cross(z, cross) / np.linalg.norm(np.cross(z, cross))
        negative_roll_cross = np.cross(cross, z) / np.linalg.norm(np.cross(cross, z))

        roll_cross_frame = np.eye(4)
        roll_cross_frame[:3, 0] = roll_cross
        roll_cross_frame[:3, 1] = np.cross(cross, roll_cross) / np.linalg.norm(np.cross(cross, roll_cross))
        roll_cross_frame[:3, 2] = cross
        roll_cross_frame[:3, 3] = np.array([self.cartesian_state.position.x, self.cartesian_state.position.y, self.cartesian_state.position.z])

        if np.dot(roll_cross, x_axis) > np.dot(negative_roll_cross, x_axis):
            # roll_cross is the nearer vector
            # print("Roll Cross is the nearer vector")
            direction = np.cross(x_axis, roll_cross) / np.linalg.norm(np.cross(x_axis, roll_cross))
            direction = np.dot(direction, cross)
            # print("Roll Direction: ", direction)
            angle = np.arccos(np.dot(x_axis, roll_cross))
            roll_angle = direction * angle
        else:
            # print("Negative Roll Cross is the nearer vector")
            # negative_roll_cross is the nearer vector
            direction = np.cross(x_axis, negative_roll_cross) / np.linalg.norm(np.cross(x_axis, negative_roll_cross))
            direction = np.dot(direction, z_axis)
            # print("Roll Direction: ", direction)
            angle = np.arccos(np.dot(x_axis, negative_roll_cross))
            roll_angle = direction * angle

        wrist_state = SimpleJointAngleCommand()
        wrist_state.q0 = -roll_angle 
        wrist_state.q1 = pitch_angle
        
        try:
            self.wrist_state_pub.publish(wrist_state)
        except rospy.ROSException as e:
            print("Failed to publish wrist state once (but should be okay)...")

        self.last_pitch_angle = pitch_angle
        self.last_roll_angle = roll_angle
        # print("Published wrist state: ", wrist_state)

    def run(self):
        
        # ensure that the spoon is always horizontal (position control)

        raise NotImplementedError
    
    def cleanup(self):
        """Clean up resources (like unsubscribing from topics, stopping publishers)."""
        if self.cartesian_state_sub:
            self.cartesian_state_sub.unregister()
        # Unregister other publishers/subscribers if needed
        print("HorizontalSpoon cleaned up!")
    

if __name__ == "__main__":
    rospy.init_node('horizontal_spoon', anonymous=True)
    hs = HorizontalSpoon()
    # hs.run()
    rospy.spin()