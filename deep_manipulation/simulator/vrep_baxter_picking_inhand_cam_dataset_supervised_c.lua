--[[ File
@description:
    This class is for constructing a manipulation simulator emtry based on vrep.
@version: V0.00
@author: Fangyi Zhang   email:gzzhangfangyi@gmail.com
@acknowledgement:
    ARC Centre of Excellence for Robotic Vision (ACRV)
    Queensland Univsersity of Technology (QUT)
@history:
    V0.00   20/04/2017  developed the first version
]]

require "simulator/class_dataset"

args = {}
-- Settings for the manipulator
-- args.joint_end_index = {3, 5, 7} -- the joint index of the controlable joints
args.vrep_app = '/home/n9314181/V-REP_PRO_EDU_V3_3_1_64_Linux/vrep.sh'
args.vrep_scene = 'vrep_scene/Baxter_Object_Picking_noKinect_vel_shape_color_rand.ttt'
args.online_mode = false
-- args.ctrl_mode = 'velocity'
-- args.ctrl_freq_t = 500
-- args.syn_mode = true
-- args.debug_mode = true
-- args.pose_constant = true
-- args.target_constant = {0.4, 0, 0.025}
-- args.left_arm_init_pose = {-0.4,0.2,-0.5,0.3,-0.8,0.6,0}
args.left_arm_init_pose = {-0.5,-1.0,0.0,1.5,0.0,0.8,0}
-- args.left_arm_init_pose = {-0.5,-0.8,0.0,1.5,0.0,0.8,0}
-- args.left_arm_init_pose = {-0.5,-0.6,0.0,1.5,0.0,0.8,0}
args.image_format = 'RGB'
-- args.image_res = {400,400}
args.image_res = {300,300}

-- Ground truth case
-- args.target_pose_max = {0.85, 0.6, 0.5708}
-- args.target_pose_min = {0.35, 0.0, 0.4208}
-- Consider the noisy table position
args.target_pose_max = {0.85, 0.6, 0.5708}
-- args.target_pose_max = {0.85, 0.6, 0.4308}
args.target_pose_min = {0.35, 0.0, 0.3908}
-- args.target_pose_min = {0.35, 0.0, 0.4108}
args.camera_pose_range = 0.02
args.table_position =  {0.635, 0.15, 0.75}
args.table_position_range = {0.03, 0.1, 0.02}
args.table_orientation = {0.0, 0.0, 1.57}
args.table_orientation_range = {0.0, 0.0, 0.14}
args.add_noise = true
args.noisy_prob = 0.8
args.upper_plane = true
-- args.target_h = 0.0325

args.weighted_losses = true

-- args.left_arm_init_pose = {0.2,-1.5,0.2,1.5,0.6,0.8,0.6}

-- initialize a reward function object
args.step_interval = 1
args.max_step = 200

args.dataset = 'control_dataset_333.t7' -- control sim dataset


-- args.dataset = 'table_top_blue_box_dataset0.01_0.t7'
-- args.dataset = 'image300_control_dataset_off_vel0.05_01.t7'
-- args.dataset = 'table_top_blue_box_dataset0.01_03.t7'
-- args.dataset = 'image300_control_dataset_off_vel0.05_01.t7'
-- args.dataset = 'image300_control_dataset_off_vel0.05_0.t7'
-- args.dataset = 'image300_control_dataset_off_vel0.03_0.t7'

-- args.dataset = 'table_top_blue_box_dataset_dr_constrained0.05_0.t7' -- perception sim dataset


-- args.extra_dataset = 'table_top_blue_box_dataset0.01_01.t7'
-- args.extra_dataset = 'table_top_blue_box_dataset_dr_constrained0.01_0.t7'
-- args.real_dataset = 'sorted_image300_real_data_37_3.t7'
-- args.real_dataset = 'sorted_image300_real_data_1t2.t7'
-- args.real_dataset = 'sorted_image300_real_data_14.t7' -- perception unlabelled real dataset

-- args.extra_dataset = 'sorted_image300_real_data_12.t7' -- perception labelled real dataset
-- args.extra_dataset = 'table_top_blue_box_dataset_dr_constrained0.05_0.t7'





return dataset_ctrl(args)
