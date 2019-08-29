#!/usr/bin/env python

# [[ File
# @description:
#     This class is for controlling Franka arm in the context of tabletop object picking.
# @version: V0.10
# @author: Fangyi Zhang   email:gzzhangfangyi@gmail.com
# @acknowledgement:
#     ARC Centre of Excellence for Robotic Vision (ACRV)
#     Queensland Univsersity of Technology (QUT)
# @history:
#     V0.00   31/08/2018  converted from baxter_vel
#     V0.10   19/09/2018  fixed the joint position problem and added a new option to control an arm to a desired joint configuration
# ]]


# import argparse
import os
import os.path

# import time
# from msvcrt import getch
import sys, tty, termios

import numpy as np
import cv2
import cv_bridge
import rospy

from std_msgs.msg import String
from sensor_msgs.msg import Image
from franka_joint_controllers.msg import JointVelocity, JointPosition
from franka_msgs.msg import FrankaState, Errors as FrankaErrors
from moveit_commander import MoveGroupCommander
from actionlib_msgs.msg import GoalStatusArray
from sensor_msgs.msg import JointState
import controller_manager_msgs.srv as cm_srv

# from ar_track_alvar_msgs.msg import AlvarMarkers
import tf
import math
import lua
# lua.require("torch")
# lg = lua.globals()



# A class for controller management
class ControlSwitcher:
    def __init__(self, controllers, controller_manager_node='/controller_manager'):
        # Dictionary of controllers to manager/switch:
        # {nick_name: controller_full_name}
        self.controllers = controllers

        rospy.wait_for_service(controller_manager_node + '/switch_controller')
        rospy.wait_for_service(controller_manager_node + '/list_controllers')
        self.switcher_srv = rospy.ServiceProxy(controller_manager_node + '/switch_controller', cm_srv.SwitchController)
        self.lister_srv = rospy.ServiceProxy(controller_manager_node + '/list_controllers', cm_srv.ListControllers)

    def switch_controller(self, controller_name):
        rospy.sleep(0.5)
        start_controllers = [self.controllers[controller_name]]
        stop_controllers = [self.controllers[n] for n in self.controllers if n != controller_name]

        controller_switch_msg = cm_srv.SwitchControllerRequest()
        controller_switch_msg.strictness = 1
        controller_switch_msg.start_controllers = start_controllers
        controller_switch_msg.stop_controllers = stop_controllers

        res = self.switcher_srv(controller_switch_msg).ok
        if res:
            rospy.loginfo('Successfully switched to controller %s (%s)' % (controller_name, self.controllers[controller_name]))
            return res
        else:
            return False

class RobotInterface(object):
    """ dqnROSinterface
        An interface for subscribe image topics and publish control command topics
    """

    def __init__(self, use_moveit=False):
        # self.is_stopping = False

        rospy.loginfo('Starting the Franka interface!')
        rospy.init_node("deeprobot_franka_interface")
        # self.rate = rospy.Rate(30) # 30 Hz
        self.rate = rospy.Rate(100) # 100 Hz
        self.curr_velo_pub = rospy.Publisher('/joint_velocity_node_controller/joint_velocity', JointVelocity, queue_size=1)
        self.curr_pos_pub = rospy.Publisher('/joint_position_node_controller/joint_position', JointPosition, queue_size=1)
        # self.pc = PandaCommander(group_name='panda_arm_hand')
        self.joint_names = {'left_s0', 'left_s1', 'left_e0', 'left_e1', 'left_w0', 'left_w1', 'left_w2'}
        self.index_corr = {'left_s0': 1, 'left_s1': 2, 'left_e0': 3, 'left_e1': 4, 'left_w0': 5, 'left_w1': 6, 'left_w2': 7} # the index correspondence


        self.bridge = cv_bridge.CvBridge()
        self.curr_im = np.zeros((3,480,640))
        self.control_accuracy = 0.01
        self.use_moveit = use_moveit
        if self.use_moveit:
            self.cs = ControlSwitcher({'moveit': 'position_joint_trajectory_controller',
                                       'velocity': 'joint_velocity_node_controller'})
            rospy.wait_for_message('move_group/status', GoalStatusArray)
            self.panda_moveit_commander = MoveGroupCommander('panda_arm')
        else:
            self.cs = ControlSwitcher({'position': 'joint_position_node_controller',
                                       'velocity': 'joint_velocity_node_controller'})
        self.cs.switch_controller('velocity')
        self.target_joint_state = JointState()
        self.target_joint_state.name = ["panda_joint1", "panda_joint2", "panda_joint3", "panda_joint4", "panda_joint5", "panda_joint6", "panda_joint7"]
        self.target_joint_state.position = [-1.29, -0.26, -0.27, -2.34, 0.12, 2.10, 0.48]
        self.robot_state = None
        self.curr_joint_angles = None
        self.ROBOT_ERROR_DETECTED = False
        self.image_sub = rospy.Subscriber("/camera/color/image_raw",Image,self.__updateCurrRGBImage)
        self.robot_state_sub = rospy.Subscriber('/franka_state_controller/franka_states', FrankaState, self.__robot_state_callback, queue_size=1)

        # magic constant: joint speed :)
        # self.interface.set_joint_position_speed(0.9)

        # if moveToInitial: self.initCam()

        # initialize the desired position with the current position
        # angles = self.interface.joint_angles()
        # self.desired_joint_pos = { key: angles[key] for key in self.joint_names }
        rospy.loginfo('Initialization finished!')

        # rospy.Subscriber("ar_pose_marker", AlvarMarkers, self.ar_pose_update)
        # self.action_implemented = True
        # rospy.spin()
        # self.rate.sleep()

    def getArmPose(self):
        rospy.wait_for_message('/franka_state_controller/franka_states', FrankaState)
        robot_pose = lua.eval("{0, 0, 0, 0, 0, 0, 0}")
        for key in self.joint_names:
            robot_pose[self.index_corr[key]] = self.curr_joint_angles[self.index_corr[key]-1]

        return robot_pose

    def goMoveit(self, desired_arm_pos):
        self.cs.switch_controller('moveit')
        self.target_joint_state.position = desired_arm_pos
        self.panda_moveit_commander.go(self.target_joint_state)
        self.cs.switch_controller('velocity')

    def goJointPosition(self, desired_arm_pos):
        self.cs.switch_controller('position')
        # Move to the desired_arm_pos
        # self.curr_pos_pub.publish(desired_arm_pos)
        while not rospy.is_shutdown():
            self.curr_pos_pub.publish(desired_arm_pos)
            rospy.wait_for_message('/franka_state_controller/franka_states', FrankaState)
            arm_pose_reached = True
            for key in self.joint_names:
                # if self.robot_state.dq[self.index_corr[key]-1] > 0.0001:
                pos_error = self.curr_joint_angles[self.index_corr[key]-1] - desired_arm_pos[self.index_corr[key]-1]
                if abs(pos_error) > self.control_accuracy or self.robot_state.dq[self.index_corr[key]-1] > 0.005:
                    arm_pose_reached = False
            # print(pos_error)
            if arm_pose_reached:
                break
        rospy.sleep(3)
        # # Wait the motion to complete
        # while not rospy.is_shutdown():
        #     rospy.wait_for_message('/franka_state_controller/franka_states', FrankaState)
        #     motion_completed = True
        #     for key in self.joint_names:
        #         if self.robot_state.dq[self.index_corr[key]-1] > 0:
        #             motion_completed = False
        #
        #     if motion_completed:
        #         break
        self.cs.switch_controller('velocity')

    def setArmPose(self, data):
        desired_arm_pos = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        for key in self.joint_names:
            desired_arm_pos[self.index_corr[key]-1] = data[self.index_corr[key]]

        rospy.loginfo('Moving the arm to a designated pose ...')

        if self.use_moveit:
            self.goMoveit(desired_arm_pos)
        else:
            self.goJointPosition(desired_arm_pos)

        rospy.loginfo('Reached the designated pose!!!')

        return self.getArmPose()

    def setArmVel(self, data):
        # rospy.loginfo('Updating the %s arm velocity ...', self.limb)
        # limb_joint_pos = { 'left_s0': -0.5, 'left_s1': data.data[0],
        #                    'left_w0': data.data[1], 'left_w1': data.data[2],
        #                    'left_w2': data.data[3], 'left_e0': data.data[4],
        #                    'left_e1': data.data[5]}
        # print(data)
        try:
            desired_arm_vel = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            for key in self.joint_names:
                desired_arm_vel[self.index_corr[key]-1] = data[self.index_corr[key]]
            self.curr_velo_pub.publish(desired_arm_vel)
        except KeyboardInterrupt:
            rospy.loginfo('Please measure the distance, then press the Enter button ...')
            self.waitKeyEnter()

        # key = ord(self.getch())
        # rospy.loginfo('Key: %s', key)
        # if key == 13:



        return

    def initHead(self, data):
        print("====================================")
        print("No need to set the head pose here for the Franka panda arm!!!")
        print("====================================")

    def setCam(self, auto):
        print("====================================")
        print("No need to set camera parameters here for the Franka panda arm!!!")
        print("====================================")


    def initCam(self, data):
        print("====================================")
        print("No need to intialize the camera here for the Franka panda arm!!!")
        print("====================================")


    def getRGBImage(self):
        # return lua.eval(self.curr_im)
        # return self.curr_im
        # print(self.curr_im.shape)
        # im = torch.fromNumpyArray(np.transpose(self.curr_im,(2, 0, 1)))
        # print(torch.isTensor(im))
        # im = torch._ByteTensor(im)
        # im = lg.torch.serialize(im)
        # im = torch.zeros(3,400,400)
        # print(torch._type(im))
        # print(im)
        # im = im._totable()
        # im = im._serialize()
        # im = np.array2string(np.transpose(self.curr_im,(2, 0, 1)))
        # for i in range(1,12):
        rospy.wait_for_message("/cameras/right_hand_camera/image",Image)
        # self.rate.sleep()
        im = np.transpose(self.curr_im,(2, 0, 1))
        # im = self.curr_im
        im = im.tobytes()
        # im.tostring()
        # im.decode('UTF-8')
        # print(im)
        return im

    def __updateCurrRGBImage(self, data):
        # tt = time.time()
        # print("Current Time: ", tt)
        # rospy[torch.LongStorage of size 1]
        # rospy.loginfo('Updating camera image ...')
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "rgb8")
        except CvBridgeError as e:
            print(e)

        # cv2.imshow("Image window", cv_image)
        self.curr_im = np.asarray(cv_image)
        # rospy.loginfo('Updated camera image ...')

        # cv2.waitKey(3)
        # print("image update time cost: ",time.time() - tt)

    def __robot_state_callback(self, data):
        self.robot_state = data
        self.curr_joint_angles = data.q
        for s in FrankaErrors.__slots__:
            if getattr(data.current_errors, s):
                self.stop()
                rospy.logerr('Robot Error Detected')
                self.ROBOT_ERROR_DETECTED = True

    def stop(self):
        # self.pc.stop()
        vel_command = Velocity()
        self.curr_velo_pub.publish(vel_command)

    def waitKeyEnter(self):
        rospy.loginfo('Please manually move the arm to a desired pose ...')
        # Wait for the press of Enter
        try:
            input('Move the arm to a desired pose, then press Enter: ')
        except SyntaxError:
            pass

    def getch(self):
        """getch() -> key character

        Read a single keypress from stdin and return the resulting character.
        Nothing is echoed to the console. This call will block if a keypress
        is not already available, but will not wait for Enter to be pressed.

        If the pressed key was a modifier key, nothing will be detected; if
        it were a special function key, it may return the first character of
        of an escape sequence, leaving additional characters in the buffer.
        """
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            ch = sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        return ch


# A function for testing
def main():
    # Initialize a RobotInterface object
    # Use the moveit control by setting the augument to True; joint position control is used by default
    # franka_arm=RobotInterface(True)
    franka_arm=RobotInterface()
    d_pos=lua.eval("{-1.29, -0.26, -0.27, -2.34, 0.12, 2.10, 0.48}")
    # d_pos=lua.eval("{-1.4, -0.56, -0.37, -2.14, 0.52, 1.80, 0.78}")

    d_vel=lua.eval("{0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1}")
    # d_vel=lua.eval("{0.0, 0., 0.0, 0.0, 0.0, 0.0, 0.0}")

    # zero_vel=lua.eval("{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}")

    franka_arm.setArmPose(d_pos)
    # while not rospy.is_shutdown():
    for x in range(0, 50000):
        franka_arm.setArmVel(d_vel)
    # franka_arm.setArmVel(zero_vel)


if __name__ == "__main__":
    main()
