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

require "simulator/class_tabletop_picking"

args = {}
-- Settings for the manipulator
-- args.joint_end_index = {3, 5, 7} -- the joint index of the controlable joints
args.vrep_app = '/home/fangyi/Documents/V-REP_PRO_EDU_V3_3_2_64_Linux/vrep.sh'
args.vrep_scene = 'vrep_scene/Baxter_Object_Picking_noKinect_vel_shape_color_rand.ttt'
args.sim_online = true
args.real_world = false
args.e2e_mode = true
args.RL_mode = false
args.ctrl_mode = 'velocity'
args.ctrl_freq_t = 50
args.syn_mode = true
args.debug_mode = true
-- args.pose_constant = true
-- args.target_constant = {0.5950, 0.1738, 0.3584}
-- args.target_constant = {0.5950, 0.2738, 0.3584}
-- args.target_constant = {0.6750, 0.4738, 0.3584}
-- args.target_constant = {0.6750, 0.4238, 0.3584}
-- args.target_constant = {0.6750, 0.2438, 0.4184}
-- args.target_constant = {0.6750, 0.4238, 0.4084}
-- args.left_arm_init_pose = {-0.4,0.2,-0.5,0.3,-0.8,0.6,0}
-- args.left_arm_init_pose = {-0.5,-1.0,0.0,1.5,0.0,0.8,0}
-- args.left_arm_init_pose = {-0.5,-0.8,0.0,1.5,0.0,0.8,0}
-- args.left_arm_init_pose = {-0.5,-0.6,0.0,1.5,0.0,0.8,0}
args.image_format = 'RGB'
-- args.image_res = {400,400}
args.image_res = {300,300}

args.left_arm_init_pose = {-0.3,-1.2,0.2,1.2,0.2,0.7,0.0}

-- The range for target regularization
-- args.target_pose_max = {0.85, 0.6, 0.5708}
-- args.target_pose_min = {0.35, 0.0, 0.4208}
-- Consider the noisy table position
args.target_pose_max = {0.85, 0.6, 0.5708}
-- args.target_pose_max = {0.85, 0.6, 0.4308}
args.target_pose_min = {0.35, 0.0, 0.3908}
-- args.target_pose_min = {0.35, 0.0, 0.4108}
-- The range defines the target region
-- args.target_max = {0.85, 0.6, 0.5708}
args.target_max = {0.85, 0.5, 0.4208}
args.target_min = {0.40, 0.05, 0.4208}
args.camera_pose_range = 0.02 -- +-0.01
args.table_position =  {0.635, 0.15, 0.75}
args.table_position_range = {0.03, 0.1, 0.02} -- +- * m
args.table_orientation = {0.0, 0.0, 1.57}
args.table_orientation_range = {0.0, 0.0, 0.14} -- +-4degree
args.add_noise = true
args.noisy_prob = 0.8
args.upper_plane = true
-- args.target_h = 0.0325

args.no_clutters = false
-- initialize a reward function object
args.step_interval = 1
args.max_step = 100
-- args.max_step = 40

args.test_dataset = 'image300_test_set_0.1.t7'
-- args.test_dataset = 'image400_test_set_0.1.t7'
args.custom_index = {1,2,3,4,5,6,7,8,9,10,12,14,16,18,20}

return tabletop_picking(args)
