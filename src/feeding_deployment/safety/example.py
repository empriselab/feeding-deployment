import rospy

import sys

sys.path.append("../../FLAIR/bite_acquisition/scripts")

from robot_controller.kinova_controller import KinovaRobotController

if __name__ == "__main__":
    rospy.init_node("robot_controller", anonymous=True)
    robot_controller = KinovaRobotController()

    in_front_mouth = [
        3.6856442400437914,
        5.691475654070293,
        4.278732611649419,
        4.156501240781943,
        6.204998093368168,
        0.2559528670343144,
        2.2013977191992713,
    ]
    before_transfer = [
        3.522011899502162,
        5.096386982169907,
        4.5817731794573175,
        4.471311116395229,
        0.2697529849736983,
        6.192221844354622,
        2.4944114855030213,
    ]

    robot_controller.set_joint_position(before_transfer)

    input("Press enter to move to in front of mouth...")
    robot_controller.set_joint_position(in_front_mouth)
