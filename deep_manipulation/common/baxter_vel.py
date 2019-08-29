#!/usr/bin/env python

# [[ File
# @description:
#     This class is for controlling Baxter in the context of tabletop object picking.
# @version: V0.03
# @author: Fangyi Zhang   email:gzzhangfangyi@gmail.com
# @acknowledgement:
#     ARC Centre of Excellence for Robotic Vision (ACRV)
#     Queensland Univsersity of Technology (QUT)
# @history:
#     V0.00   01/06/2017  developed the first version
#     V0.02   02/06/2017  added the function to grab images through ros topics
#     V0.03   03/06/2017  fixed the bug of no response time for image subscriber
# ]]


import argparse
import rospy
import baxter_interface
import baxter_external_devices
import os
import os.path

# import time
# from msvcrt import getch
import sys, tty, termios


import numpy as np
import cv2
import cv_bridge


from baxter_interface import CHECK_VERSION
from std_msgs.msg import String
from sensor_msgs.msg import Image

# from ar_track_alvar_msgs.msg import AlvarMarkers
import tf
import math
import lua
# lua.require("torch")
# lg = lua.globals()
# import lutorpy
# lutorpy.LuaRuntime(zero_based_index=False)
# require("torch")

class RobotInterface(object):
    """ RobotInterface
        An interface for subscribe image topics and publish control command topics
    """

    def __init__(self, limb='left', moveToInitial=True, device=0):
        # self.is_stopping = False

        rospy.loginfo('Starting the 3DOF controller for the %s limb', limb)
        rospy.init_node("dqn_baxter")
        self.rate = rospy.Rate(30) # 10 Hz

        self.limb = limb
        self.joint_names = {'left_s0', 'left_s1', 'left_e0', 'left_e1', 'left_w0', 'left_w1', 'left_w2'}
        self.index_corr = {'left_s0': 1, 'left_s1': 2, 'left_e0': 3, 'left_e1': 4, 'left_w0': 5, 'left_w1': 6, 'left_w2': 7} # the index correspondence
        self.cam_joint_names = {'right_s0', 'right_s1', 'right_e0', 'right_e1', 'right_w0', 'right_w1', 'right_w2'}
        self.cam_index_corr = {'right_s0': 1, 'right_s1': 2, 'right_e0': 3, 'right_e1': 4, 'right_w0': 5, 'right_w1': 6, 'right_w2': 7}

        # self.joint_names = { 'left_w1', 'left_e1', 'left_s1' }
        # self.delta_angle = 0.04  # Baxter joint angles are in radians
        #self.delta_angle = 0.1  # used in the last test


        # active Baxter
        rs = baxter_interface.RobotEnable(CHECK_VERSION)
        init_state = rs.state().enabled
        self.interface = baxter_interface.Limb(limb)
        self.camera_arm = baxter_interface.Limb('right')
        self.cam = baxter_interface.CameraController('right_hand_camera')
        self.head = baxter_interface.Head()

        self.bridge = cv_bridge.CvBridge()
        self.curr_im = np.zeros((3,480,640))
        self.image_sub = rospy.Subscriber("/cameras/right_hand_camera/image",Image,self.updateCurrRGBImage)

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
        angles = self.interface.joint_angles()
        robot_pose = lua.eval("{0, 0, 0, 0, 0, 0, 0}")
        for key in self.joint_names:
            robot_pose[self.index_corr[key]] = angles[key]

        return robot_pose

    def setArmPose(self, data):
        rospy.loginfo('Moving the %s arm to a designated pose ...', self.limb)
        # limb_joint_pos = { 'left_s0': -0.5, 'left_s1': data.data[0],
        #                    'left_w0': data.data[1], 'left_w1': data.data[2],
        #                    'left_w2': data.data[3], 'left_e0': data.data[4],
        #                    'left_e1': data.data[5]}
        # print(data)
        desired_arm_pos = { key: data[self.index_corr[key]] for key in self.joint_names }
        self.interface.move_to_joint_positions(desired_arm_pos, timeout=15.0)

        return self.getArmPose()

    def setArmVel(self, data):
        # rospy.loginfo('Updating the %s arm velocity ...', self.limb)
        # limb_joint_pos = { 'left_s0': -0.5, 'left_s1': data.data[0],
        #                    'left_w0': data.data[1], 'left_w1': data.data[2],
        #                    'left_w2': data.data[3], 'left_e0': data.data[4],
        #                    'left_e1': data.data[5]}
        # print(data)
        try:
            desired_arm_vel = { key: data[self.index_corr[key]] for key in self.joint_names }
            self.interface.set_joint_velocities(desired_arm_vel)
        except KeyboardInterrupt:
            rospy.loginfo('Please measure the distance, then press the Enter button ...')
            self.waitKeyEnter()

        # key = ord(self.getch())
        # rospy.loginfo('Key: %s', key)
        # if key == 13:



        return

    def initHead(self, data):
        self.head.set_pan(data)

    def setCam(self, auto):
        # Camera parameters
        cam_resolution = (640,400)
        cam_fps = 30 # Camera frames per second
        if auto:
            cam_exposure = -1 # Camera Exposure.  Valid range is 0-100 or CameraController.CONTROL_AUTO
            # cam_exposure = 60 # Camera Exposure.  Valid range is 0-100 or CameraController.CONTROL_AUTO
            cam_gain = -1 # Camera gain.  Range is 0-79 or CameraController.CONTROL_AUTO

            cam_white_balance_red = -1 # White balance red.  Range is 0-4095 or CameraController.CONTROL_AUTO
            cam_white_balance_green = -1 # White balance green.  Range is 0-4095 or CameraController.CONTROL_AUTO
            cam_white_balance_blue = -1 # White balance blue.  Range is 0-4095 or CameraController.CONTROL_AUTO
        else:
            cam_exposure = 40 # Camera Exposure.  Valid range is 0-100 or CameraController.CONTROL_AUTO
            # cam_exposure = 60 # Camera Exposure.  Valid range is 0-100 or CameraController.CONTROL_AUTO
            cam_gain = 1 # Camera gain.  Range is 0-79 or CameraController.CONTROL_AUTO

            cam_white_balance_red = 2500 # White balance red.  Range is 0-4095 or CameraController.CONTROL_AUTO
            cam_white_balance_green = 2300 # White balance green.  Range is 0-4095 or CameraController.CONTROL_AUTO
            cam_white_balance_blue = 3550 # White balance blue.  Range is 0-4095 or CameraController.CONTROL_AUTO
            # cam_white_balance_red = 2000 # White balance red.  Range is 0-4095 or CameraController.CONTROL_AUTO
            # cam_white_balance_green = 1800 # White balance green.  Range is 0-4095 or CameraController.CONTROL_AUTO
            # cam_white_balance_blue = 2800 # White balance blue.  Range is 0-4095 or CameraController.CONTROL_AUTO

        print("Setting Camera Parameters ... ")
        if self.cam.resolution != cam_resolution:
            self.cam.resolution = cam_resolution
        if self.cam.exposure != cam_exposure:
            self.cam.exposure = cam_exposure # Camera Exposure.  Valid range is 0-100 or CameraController.CONTROL_AUTO
        if self.cam.gain != cam_gain:
            self.cam.gain = cam_gain # Camera gain.  Range is 0-79 or CameraController.CONTROL_AUTO
        if self.cam.white_balance_red != cam_white_balance_red:
            self.cam.white_balance_red = cam_white_balance_red # White balance red.  Range is 0-4095 or CameraController.CONTROL_AUTO
        if self.cam.white_balance_green != cam_white_balance_green:
            self.cam.white_balance_green = cam_white_balance_green # White balance green.  Range is 0-4095 or CameraController.CONTROL_AUTO
        if self.cam.white_balance_blue != cam_white_balance_blue:
            self.cam.white_balance_blue = cam_white_balance_blue # White balance blue.  Range is 0-4095 or CameraController.CONTROL_AUTO
        if self.cam.fps != cam_fps:
            self.cam.fps = cam_fps # Camera frames per second

        print("Resolution: ", self.cam.resolution)
        print("Explosure: ", self.cam.exposure)
        print("Gain: ", self.cam.gain)
        print("White Balance Red: ", self.cam.white_balance_red)
        print("White Balance Green: ", self.cam.white_balance_green)
        print("White Balance Blue: ", self.cam.white_balance_blue)
        print("====================================")

    def initCam(self, data):
        rospy.loginfo('Moving the right arm to a designated pose ...')
        # limb_joint_pos = { 'left_s0': -0.5, 'left_s1': data.data[0],
        #                    'left_w0': data.data[1], 'left_w1': data.data[2],
        #                    'left_w2': data.data[3], 'left_e0': data.data[4],
        #                    'left_e1': data.data[5]}
        # print(data)
        desired_cam_pos = { key: data[self.cam_index_corr[key]] for key in self.cam_joint_names }
        # self.camera_arm.move_to_joint_positions(desired_cam_pos, timeout=10.0)
        self.camera_arm.move_to_joint_positions(desired_cam_pos, timeout=15.0)
        # self.camera_arm.move_to_joint_positions(desired_cam_pos, timeout=15.0,  threshold=0.0001)
        angles = self.camera_arm.joint_angles()
        print("Camera Arm Pose: ",angles)

        self.setCam(False)

        # print("Setting Camera Parameters ... ")

        # self.cam.resolution = (640,400)
        # self.cam.exposure = 40 # Camera Exposure.  Valid range is 0-100 or CameraController.CONTROL_AUTO
        # # self.cam.exposure = 60 # Camera Exposure.  Valid range is 0-100 or CameraController.CONTROL_AUTO
        # self.cam.gain = 1 # Camera gain.  Range is 0-79 or CameraController.CONTROL_AUTO
        #
        # # self.cam.white_balance_red = 1500 # White balance red.  Range is 0-4095 or CameraController.CONTROL_AUTO
        # # self.cam.white_balance_green = 1000 # White balance green.  Range is 0-4095 or CameraController.CONTROL_AUTO
        # # self.cam.white_balance_blue = 2000 # White balance blue.  Range is 0-4095 or CameraController.CONTROL_AUTO
        #
        # # Current best
        # self.cam.white_balance_red = 2500 # White balance red.  Range is 0-4095 or CameraController.CONTROL_AUTO
        # self.cam.white_balance_green = 2300 # White balance green.  Range is 0-4095 or CameraController.CONTROL_AUTO
        # self.cam.white_balance_blue = 3550 # White balance blue.  Range is 0-4095 or CameraController.CONTROL_AUTO
        #
        # # self.cam.white_balance_red = 2000 # White balance red.  Range is 0-4095 or CameraController.CONTROL_AUTO
        # # self.cam.white_balance_green = 1800 # White balance green.  Range is 0-4095 or CameraController.CONTROL_AUTO
        # # self.cam.white_balance_blue = 2800 # White balance blue.  Range is 0-4095 or CameraController.CONTROL_AUTO
        #
        # self.cam.fps = 30 # Camera frames per second
        # print("====================================")
        # print("Resolution: ", self.cam.resolution)
        # print("Explosure: ", self.cam.exposure)
        # print("Gain: ", self.cam.gain)
        # print("White Balance Red: ", self.cam.white_balance_red)
        # print("White Balance Green: ", self.cam.white_balance_green)
        # print("White Balance Blue: ", self.cam.white_balance_blue)

        return

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

    def updateCurrRGBImage(self, data):
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

    def waitKeyEnter(self):
        rospy.loginfo('Please manually move the %s limb to a desired pose ...', self.limb)
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
