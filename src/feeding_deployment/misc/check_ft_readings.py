
import rospy
from geometry_msgs.msg import WrenchStamped
import numpy as np

def ft_callback(msg):

    ft_reading = np.array([msg.wrench.force.x, msg.wrench.force.y, msg.wrench.force.z, msg.wrench.torque.x, msg.wrench.torque.y, msg.wrench.torque.z])
    down_torque = ft_reading[3]
    # mag = np.linalg.norm(ft_reading)
    # print(f"FT reading: {ft_reading}, magnitude: {mag}")
    if np.abs(down_torque) > 0.05:
        print("Bite detected with down torque: ", down_torque)

if __name__ == '__main__':
    rospy.init_node('check_ft_readings')
    
    np.set_printoptions(precision=2, suppress=True)
    ft_sensor_sub = rospy.Subscriber('/forque/forqueSensor', WrenchStamped, ft_callback)
    rospy.spin()