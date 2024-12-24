
import numpy as np
from scipy.spatial.transform import Rotation

try:
    import rospy
    import tf2_ros
    from geometry_msgs.msg import Pose, PoseStamped
    from geometry_msgs.msg import Pose, TransformStamped
    ROSPY_IMPORTED = True
except ModuleNotFoundError as e:
    # print(f"ROS not imported: {e}")
    ROSPY_IMPORTED = False


class TFUtils:
    def __init__(self):
        self.tfBuffer = tf2_ros.Buffer() # Using default cache time of 10 secs
        self.listener = tf2_ros.TransformListener(self.tfBuffer)
        self.broadcaster = tf2_ros.TransformBroadcaster()
        self.control_rate = rospy.Rate(100)
    
    def getTransformationFromTF(self, source_frame, target_frame):

        while not rospy.is_shutdown():
            try:
                # print(f"Looking for transform from {source_frame} to {target_frame} using tfBuffer.lookup_transform...")
                transform = self.tfBuffer.lookup_transform(source_frame, target_frame, rospy.Time())
                # print("Got transform!")
                break
            except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
                self.control_rate.sleep()
                continue

        T = np.zeros((4,4))
        T[:3,:3] = Rotation.from_quat([transform.transform.rotation.x, transform.transform.rotation.y, transform.transform.rotation.z, transform.transform.rotation.w]).as_matrix()
        T[:3,3] = np.array([transform.transform.translation.x, transform.transform.translation.y, transform.transform.translation.z]).reshape(1,3)
        T[3,3] = 1

        # print("Translation: ", T[:3,3])
        # print("Rotation in quaternion: ", transform.transform.rotation.x, transform.transform.rotation.y, transform.transform.rotation.z, transform.transform.rotation.w)
        # print("Rotation in euler: ", Rotation.from_quat([transform.transform.rotation.x, transform.transform.rotation.y, transform.transform.rotation.z, transform.transform.rotation.w]).as_euler('xyz', degrees=True))

        return T
    
    def publishTransformationToTF(self, source_frame, target_frame, transform):

        t = TransformStamped()

        t.header.stamp = rospy.Time.now()
        t.header.frame_id = source_frame
        t.child_frame_id = target_frame

        t.transform.translation.x = transform[0][3]
        t.transform.translation.y = transform[1][3]
        t.transform.translation.z = transform[2][3]

        R = Rotation.from_matrix(transform[:3,:3]).as_quat()
        t.transform.rotation.x = R[0]
        t.transform.rotation.y = R[1]
        t.transform.rotation.z = R[2]
        t.transform.rotation.w = R[3]

        self.broadcaster.sendTransform(t)

    def get_pose_msg_from_transform(self, transform):

        pose = Pose()
        pose.position.x = transform[0,3]
        pose.position.y = transform[1,3]
        pose.position.z = transform[2,3]

        quat = Rotation.from_matrix(transform[:3,:3]).as_quat()
        pose.orientation.x = quat[0]
        pose.orientation.y = quat[1]
        pose.orientation.z = quat[2]
        pose.orientation.w = quat[3]

        return pose
