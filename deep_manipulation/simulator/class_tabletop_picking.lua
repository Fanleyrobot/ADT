--[[ File
@description:
    This class is for contructing a manipulation simulator based on VREP.
@version: V0.20
@author: Fangyi Zhang   email:gzzhangfangyi@gmail.com
@acknowledgement:
    ARC Centre of Excellence for Robotic Vision (ACRV)
    Queensland Univsersity of Technology (QUT)
@history:
    V0.00   18/04/2017  developed the first version
    V0.01   19/05/2017  added a function (getTargetConfig) for perception test with guidance controller
    V0.10   25/05/2017  optimized the position control mode codes
    V0.11   25/05/2017  added a function for velocity control
    V0.12   30/05/2017  added functions to get batches for control
    V0.13   02/06/2017  added functions to grab images from ros topics
    V0.14   03/06/2017  added a function for grab one real image
    V0.15   05/06/2017  enabled importing baxter_vel from ./common
    V0.16   14/06/2017  updated according to the changes in simulator class
    V0.20   21/09/2018  made compatible to both baxter and franka arms
]]


require 'torch'
-- require 'gnuplot'
require 'image'
require 'common/interface_vrep'
require 'reward/reward_functions_picking'
-- local interface_vrep = require 'common/interface_vrep'
-- local reward_functions_picking = require 'reward/reward_functions_picking'

-- construct a class
local sim = torch.class('tabletop_picking')

--[[ Function
@description: initialize an object for manipulation simulation
@input:
    args: settings for a manipulation simulation object
@output: nil
@notes:
]]
function sim:__init(args)
    -- Primary Settings
    self.image_format = args.image_format or 'Grey'
    self.image_res = args.image_res or {300,300}
    self.sim_online = args.sim_online or false
    self.real_world = args.real_world or false
    self.syn_mode = args.syn_mode or false
    self.e2e_mode = args.e2e_mode or false
    self.RL_mode = args.RL_mode or false
    self.sim_step = args.sim_step or 50 -- 50 ms
    self.max_step = args.max_step or 400

    self.left_joint_position = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}
    self.right_joint_position = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}
    self.left_joint_velocity = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}
    self.right_joint_velocity = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}

    -- Settings for Clutters
    self.clutters = args.clutters or {'Baseball','Block','Crayons','Dove','Eraser','EraserPack','Note1','RubCube','UQBall'}
    self.clutters_h = args.clutters_h or {0.0375,0.01,0.0125,0.0175,0.03,0.0175,0.005,0.029,0.03}
    self.n_clutters = #self.clutters
    self.no_clutters = args.no_clutters or false

    -- Initialize vrep
    print("Sim_onlne:",self.sim_online)
    if self.sim_online then
      self:initVREP(args)
    end


    if args.test_dataset then
      -- Load pre-generated datasets
      self.test_dataset = self:load_pre_constructed_dataset(args.test_dataset)
      -- Decide whether the dataset satisfies current accuracy requirements
    --   self.test_dataset_image = self.test_dataset.image
      self.test_dataset_target = self.test_dataset.object_position
      self.test_dataset_target_arm_pose = self.test_dataset.target_arm_pose
      self.test_dataset_arm_pose = self.test_dataset.arm_pose
      self.test_dataset_sample_amount = self.test_dataset.sample_amount
      self.testset_index = 0
      self.customized_index = args.custom_index or nil
      if self.customized_index then
          print("custom_index", torch.Tensor(self.customized_index))
          self.test_dataset_sample_amount = #self.customized_index
      end

      io.flush()
      collectgarbage()
    end

    -- Task parameters
    self.ctrl_mode = args.ctrl_mode or 'position'
    print("Control Mode: ", self.ctrl_mode)
    self.ctrl_freq_t = args.ctrl_freq_t or 50 -- 50 ms
    self.n_sim_one_ctrl = math.floor(self.ctrl_freq_t / self.sim_step)
    self.target_constant = args.target_constant or nil
    self.pose_constant = args.pose_constant or false
    self.target_region_x = 0.725
    self.target_region_y = 0.8
    self.safe_edge = 0.3 --0.05
    self.target_z = 0.025

    self.left_arm_init_pose = args.left_arm_init_pose or {-0.4,0.2,-0.5,0.3,-0.8,0.6,0} -- initial joint pose of an arm (rad)
    self.right_arm_init_pose = args.right_arm_init_pose or {-0.3,-1.3,1,1.1,0.4,1.8,-1.57} -- initial joint pose of an arm (rad)
    self.n_joint = args.n_joint or #self.left_arm_init_pose
    self.action_step = args.action_step or 0.04 -- joint angle changing rate in each step, set it to -0.04, because the angle sign is opposite in Baxter
    self.vel_const = args.vel_const or 0.5
    -- self.target, self.left_joint_position = self:initScene()

    self.joint_pose_max = args.joint_pose_max or {1.702, 1.048, 3.055, 2.619, 3.060, 2.095, 3.060}
    self.joint_pose_min = args.joint_pose_min or {-1.702, -2.148, -3.055, -0.06, -3.060, -1.571, -3.060}
    -- The range for target value regularization
    self.target_pose_max = args.target_pose_max or {1.019, 0.6617, 0.4584}
    self.target_pose_min = args.target_pose_min or {0.419, 0.1367, 0.3084}
    -- The range defines the target region
    self.target_max = args.target_max or {1.019, 0.6617, 0.4584}
    self.target_min = args.target_min or {0.419, 0.1367, 0.3084}
    self.camera_pose_range = args.camera_pose_range or 0.02
    self.table_position = args.table_position or  {0.635, 0.15, 0.75}
    self.table_position_range = args.table_position_range or {0.03, 0.1, 0.02}
    self.table_orientation = args.table_orientation or {0.0, 0.0, 1.57}
    self.table_orientation_range = args.table_orientation_range or {0.0, 0.0, 0.14}

    self.step_interval = args.step_interval or 1
    self.max_step = args.max_step or 100
    self.used_steps = 0
    if self.RL_mode then
      -- Initialize a reward function object
      self.rwd_ob = reward_functions_picking{step_interval=self.step_interval, max_step=self.max_step}
    end

    -- Training Settings
    self.add_noise = args.add_noise or false
    self.noisy_prob = args.noisy_prob or 1
    self.upper_plane = args.upper_plane or false
    self.target_h = args.target_h or 0.0325
    if self.upper_plane then
      self.target_pose_min[3] = self.target_pose_min[3] + self.target_h
      self.target_pose_max[3] = self.target_pose_max[3] + self.target_h
    end

    -- Evaluation Settings
    self.closest_distance = 100
    self.completion = false

    -- Initialize ROS node and Baxter
    if self.real_world then
      require 'python'
      self.rospy = python.import("rospy")
      -- Added the common folder into the python path for baxter_vel or franka_vel import
      python.execute("import sys")
      python.execute("sys.path.insert(0, './common')")
      self.inter_class = args.inter_class or "franka_vel"
      print("Real robot interface: ", self.inter_class)
      self.ROSinter = python.import(self.inter_class)
      -- self.inter = self.ROSinter.RobotInterface('left')
      self.inter = self.ROSinter.RobotInterface()
      self.inter.initHead(0.53)
      -- self.inter.initCam({0,-1.6,1,0.9,0.3,2.6,0})
      self.inter.initCam(self.right_arm_init_pose)
    elseif self.sim_online then
      self:setArmTargetPose('right',self.right_arm_init_pose)
    end

    self:report()

end


-- Print parameters
function sim:report()
  print("=========== Parameters ============")
  print("Camera Arm Pose: ", torch.Tensor(self.right_arm_init_pose))
  print("Left Arm Initial Pose: ", torch.Tensor(self.left_arm_init_pose))
  print("Image Format: ", self.image_format)
  print("Image Resolution: ", torch.Tensor(self.image_res))
  print("Target Min: ", torch.Tensor(self.target_pose_min))
  print("Target Max: ", torch.Tensor(self.target_pose_max))
  print("Camera Pose Range: ", self.camera_pose_range)
  print("Table Position: ", torch.Tensor(self.table_position))
  print("Table Orientation: ", torch.Tensor(self.table_orientation))
  print("Table Position Range: ", torch.Tensor(self.table_position_range))
  print("Table Orientation Range: ", torch.Tensor(self.table_orientation_range))
  print("Whether Add Noise: ", self.add_noise)
  print("Noise Prob: ", self.noisy_prob)
  print("Whether Use Upper Plane: ", self.upper_plane)
  print("Target Height: ", self.target_h)
  print("Max Step: ", self.max_step)
  print("===================================")
end

-- Initialize vrep
function sim:initVREP(args)
  self.vrep = interface_vrep{vrep_app=args.vrep_app,
                        vrep_scene=args.vrep_scene,
                        debug_mode=args.debug_mode,
                        syn_mode=self.syn_mode,
                        connTimeout=-20000}

  -- Get object handle
  self:getObjectHandles()

  -- Get distancd handle
  self:getDistanceHandles()

  -- Get collision handle
  self:getCollisionHandles()

  -- Init Cameras
  self:initStreams()

  -- Get arm states
  self:getArmPose()
end

-- Get object handles
function sim:getObjectHandles()

  -- Cameras
  -- self.head_kinect_rgb = self:getObjectHandle('kinect_rgb_head')
  -- self.head_kinect_depth = self:getObjectHandle('kinect_depth_head')
  -- self.right_kinect_rgb = self:getObjectHandle('kinect_rgb')
  -- self.right_kinect_depth = self:getObjectHandle('kinect_depth')
  self.head_camera = self.vrep:getObjectHandle('Baxter_camera')
  self.right_camera = self.vrep:getObjectHandle('Baxter_rightArm_camera')
  self.left_camera = self.vrep:getObjectHandle('Baxter_leftArm_camera')

  -- Arms
  self.left_joint_handle = {}
  self.right_joint_handle = {}
  for i=1,7 do
    self.left_joint_handle[i] = self.vrep:getObjectHandle('Baxter_leftArm_joint' .. i)
    self.right_joint_handle[i] = self.vrep:getObjectHandle('Baxter_rightArm_joint' .. i)
  end

  -- Objects
  self.table_top = self.vrep:getObjectHandle('customizableTable_tableTop')
  self.table = self.vrep:getObjectHandle('customizableTable')
  self.cuboid = self.vrep:getObjectHandle('Cuboid')
  self.robot_base = self.vrep:getObjectHandle('Baxter_base_visible')
  self.left_gripper = self.vrep:getObjectHandle('BaxterVacuumCup_link')

  self.clutters_handles = {}
  for i=1,self.n_clutters do
    self.clutters_handles[i] = self.vrep:getObjectHandle(self.clutters[i])
  end
  -- self.ball = self.vrep:getObjectHandle('Sphere')
  -- self.cup = self.vrep:getObjectHandle('Cup')

end

-- Get collision handles
function sim:getCollisionHandles()
  self.collision_arm_handle = self.vrep:getCollisionHandle('Collision_leftArm')
  self.collision_cuboid_handle = self.vrep:getCollisionHandle('Collision_cuboid')
end

-- Get distance handles
function sim:getDistanceHandles()
  self.distance_e2t_handle = self.vrep:getDistanceHandle('Distance_e2t')
end

-- Init streaming signals
function sim:initStreams()
  local rs = self.vrep:initDistanceDetect(self.distance_e2t_handle)
  rs = self.vrep:initCollisionDetect(self.collision_arm_handle)
  rs = self.vrep:initCollisionDetect(self.collision_cuboid_handle)
  if self.image_format == 'RGB' then
    rs = self.vrep:initRGBCamera(self.right_camera)
  else
    rs = self.vrep:initGreyCamera(self.right_camera)
  end
end

-- Get arm pose
function sim:getArmPose(arm)
  if arm == 'left' then
    for i=1,7 do
      self.left_joint_position[i] = self.vrep:getJointPosition(self.left_joint_handle[i])
    end
    return self.left_joint_position
  elseif arm == 'right' then
    for i=1,7 do
      self.right_joint_position[i] = self.vrep:getJointPosition(self.right_joint_handle[i])
    end
    return self.right_joint_position
  else
    for i=1,7 do
      self.left_joint_position[i] = self.vrep:getJointPosition(self.left_joint_handle[i])
      self.right_joint_position[i] = self.vrep:getJointPosition(self.right_joint_handle[i])
    end
    return self.left_joint_position, self.right_joint_position
  end
end

-- Set desired arm pose
function sim:setArmTargetPose(arm, pos)
  if arm == 'left' then
    for i=1,7 do
      self.vrep:setJointPosition(self.left_joint_handle[i], pos[i])
    end
  elseif arm == 'right' then
    for i=1,7 do
      self.vrep:setJointPosition(self.right_joint_handle[i], pos[i])
    end
  else
    for i=1,7 do
      self.vrep:setJointPosition(self.left_joint_handle[i], pos[i])
      self.vrep:setJointPosition(self.right_joint_handle[i], pos[i])
    end
  end
end

-- Get the velocity of all joints in an arm
function sim:getArmVelocity(arm)
  if arm == 'left' then
    for i=1,7 do
      self.left_joint_velocity[i] = self.vrep:getJointVelocity(self.left_joint_handle[i])
    end
    return self.left_joint_velocity
  elseif arm == 'right' then
    for i=1,7 do
      self.right_joint_velocity[i] = self.vrep:getJointVelocity(self.right_joint_handle[i])
    end
    return self.right_joint_velocity
  else
    for i=1,7 do
      self.left_joint_velocity[i] = self.vrep:getJointVelocity(self.left_joint_handle[i])
      self.right_joint_velocity[i] = self.vrep:getJointVelocity(self.right_joint_handle[i])
    end
    return self.left_joint_velocity, self.right_joint_velocity
  end
end

-- Set target velocities of all joints in an arm
function sim:setArmVelocity(arm, vel)
  if arm == 'left' then
    for i=1,7 do
      self.vrep:setJointVelocity(self.left_joint_handle[i], vel[i])
    end
  elseif arm == 'right' then
    for i=1,7 do
      self.vrep:setJointVelocity(self.right_joint_handle[i], vel[i])
    end
  else
    for i=1,7 do
      self.vrep:setJointVelocity(self.left_joint_handle[i], vel[i])
      self.vrep:setJointVelocity(self.right_joint_handle[i], vel[i])
    end
  end
end

-- Collect monitoring information
function sim:getMeasurements()
  self.distance_e2t = self.vrep:detectDistance(self.distance_e2t_handle)
  self:getArmPose('left')
  -- self.vrep:getArmVelocity('left')
  local collision = false
  local collision_, reached = self:detectCollisions()
  if collision_ and not reached then
    collision = true
  end
  return collision, reached
end

-- Detect the termination of one trial and whether a target is reached
function sim:detectCollisions()
    local target_reached = false
    local terminate = false
    self.collision_arm = self.vrep:detectCollision(self.collision_arm_handle)
    if self.collision_arm then
      terminate = true
      self.collision_cub = self.vrep:detectCollision(self.collision_cuboid_handle)
      -- self.distance_e2t = self.vrep:detectDistance(self.distance_e2t_handle)
      if self.collision_cub then
        target_reached = true
      end
    end
    -- print(terminate, target_reached)
    return terminate, target_reached
end

-- One control step
function sim:oneControlStep()
  -- detection
  local collision = false
  local reached =false
  if self.syn_mode then
    for i=1,self.n_sim_one_ctrl do
      self.vrep:stepsForward(1)
      -- print("Forard 1 step!!!!")
      -- if detect_collision then
        local collision_, reached_ = self:detectCollisions()
        if collision_ then
          if reached_ then
            reached = true
          else
            collision = true
          end
        end

      -- end
      -- print("Simulation Time: ", self.vrep:getSimTime())
    end
  end

  return collision, reached

end

-- Speed control mode
function sim:ctrlSpeedContinuous(vel)
  if self.real_world then
    self.inter.setArmVel(vel)
    self.left_joint_position = self.inter.getArmPose()
  else
    self:setArmVelocity('left', vel)
    return self:oneControlStep()
  end
end

-- Position control mode
function sim:ctrlPosition(action)
  -- print("Action: ", action)
  local action_joint = math.floor((action-1) / 3) + 1
  local action_mov = (action - 1) % 3
  --print("action_mov:", action_mov)
  if action_joint <= self.n_joint then
      -- print(self.left_joint_position[action_joint])
      self.left_joint_position[action_joint] = self.left_joint_position[action_joint] + (action_mov - 1) * self.action_step
      self.vrep:setJointPosition(self.left_joint_handle[action_joint], self.left_joint_position[action_joint])
      -- Wait the arm gets to the desired pose
      self:oneControlStep()
      local vel = self.vrep:getJointVelocity(self.left_joint_handle[action_joint])
      -- local sum_vel = torch.Tensor(vel):abs():sum()
      while vel > 0.0001 do
        -- print("speed: ", vel)
        self:oneControlStep()
        -- vel = self:getArmVelocity('left')
        vel = self.vrep:getJointVelocity(self.left_joint_handle[action_joint])
        -- sum_vel = torch.Tensor(vel):abs():sum()
      end
  else
    print("Unknown action command for position control mode!!! Command: ", action)
  end

end

-- Position control mode 15
function sim:ctrlPosition_15(action)
  -- print("Action: ", action)
  local action_joint = math.floor((action-1) / 2) + 1
  local action_mov = (action - 1) % 2 * 2
  --print("action_mov:", action_mov)
  if action_joint <= self.n_joint then
      -- print(self.left_joint_position[action_joint])
      self.left_joint_position[action_joint] = self.left_joint_position[action_joint] + (action_mov - 1) * self.action_step
      self.vrep:setJointPosition(self.left_joint_handle[action_joint], self.left_joint_position[action_joint])
      -- Wait the arm gets to the desired pose
      self:oneControlStep()
      -- local vel = self:getArmVelocity('left')
      local vel = self.vrep:getJointVelocity(self.left_joint_handle[action_joint])
      -- local sum_vel = torch.Tensor(vel):abs():sum()
      while vel > 0.0001 do
        -- print("speed: ", vel)
        self:oneControlStep()
        -- vel = self:getArmVelocity('left')
        vel = self.vrep:getJointVelocity(self.left_joint_handle[action_joint])
        -- sum_vel = torch.Tensor(vel):abs():sum()
      end
  elseif  action_joint ~= self.n_joint + 1 then
  --   print("Stay in current pose: ", action)
  -- else
    print("Unknown action command for position control mode!!! Command: ", action)
  end
end

-- Discrete speed control mode
function sim:ctrlSpeedDiscrete(action)
  local action_joint = math.floor((action-1) / 3) + 1
  local action_mov = (action - 1) % 3
  --print("action_mov:", action_mov)
  if action_joint <= self.n_joint then
      -- print(self.left_joint_position[action_joint])
      local vel = (action_mov - 1) * self.vel_const
      self.vrep:setJointVelocity(self.left_joint_handle[action_joint], vel)
      print("Velocit Set: ", action_joint, vel)
  else
    print("Unknown action command for position control mode!!! Command: ", action)
  end
end


function sim:initScene()
  local object_position, target_arm_pose, arm_pose
  if self.test_dataset then
    if self.testset_index < self.test_dataset_sample_amount then
      self.testset_index = self.testset_index + 1
    end
    local curr_index = self.testset_index
    if self.customized_index ~= nil then
        curr_index = self.customized_index[curr_index]
    end
    print("Cumstomized Index: ",curr_index)
    object_position = table.copy(self.test_dataset_target[curr_index])
    -- print("Object Position: ", torch.Tensor(object_position))
    -- object_position = {0,0,0}
    target_arm_pose = table.copy(self.test_dataset_target_arm_pose[curr_index])
    arm_pose = table.copy(self.test_dataset_arm_pose[curr_index])
    -- arm_pose = table.copy(self.left_arm_init_pose)
  end

  if self.real_world then
    return self:initScene_real(object_position, target_arm_pose, arm_pose)
  else
    return self:initScene_sim(object_position, arm_pose)
  end
end

-- Initialize a random scene
function sim:initScene_real(target, target_arm_pose, pose)

  if not target then
    target = self:generateRandTarget()
  else
    if self.e2e_mode then
        self.inter.setArmPose(target_arm_pose)
        print(">>>>>>>>>>>>>>>>>>>> Put the blue cube to the end-effector position with random orientation !!!!!!!!!!!!!!!!!!!!!")
        self.inter.waitKeyEnter()
    end
    -- self.left_joint_position = self.inter.getArmPose()
  end

  if not pose then
    pose = self.inter.setArmPose(self.left_arm_init_pose)
  else
    pose = self.inter.setArmPose(pose)
  end

  if self.sim_online then
      -- -- Randomize target orientation
      -- local target_orien = self:randTargetOrientation()
      -- self.vrep:setObjectOrientation(self.cuboid, self.robot_base, target_orien)
      self.vrep:setObjectPosition(self.cuboid, self.robot_base, target)
      self:setArmTargetPose('left',pose)
  end

  return target, pose
end

-- Initialize a random scene
function sim:initScene_sim(target, pose)
  -- local rs = self.vrep:setImageRendering(false)
  if self.ctrl_mode ~= 'position' then
    self.vrep:enableJointPositionCtrl()
  end
  if not target then
    target = self:generateRandTarget()
  end

  -- target = self:setTarget(target)
  -- self:setCameraPose(true)
  -- self:randTablePose(true)
  self:randClutters()
  -- local table_position = self.vrep:getObjectPosition(self.table_top, self.robot_base)
  -- -- print("Table Position: ", torch.Tensor(table_position))
  -- if target[3] < table_position[3] + 0.0325 then
  --   target[3] = table_position[3] + 0.0325
  -- end
  local target_orien = self:randTargetOrientation()
  self.vrep:setObjectOrientation(self.cuboid, self.robot_base, target_orien)
  self.vrep:setObjectPosition(self.cuboid, self.robot_base, target)
  if not pose then
    pose = self:initArmPose()
    -- local desired_pose
    local desired_pose = self.vrep:getDesiredConfig(0.2)
    -- pose = self:generateRandPose(pose, desired_pose)
    if desired_pose~=nil then
      if #desired_pose~=0 then
        pose = self:generateRandPose(pose, desired_pose)
      end
    end
  else
    self:setArmTargetPose('left',pose)
    for i=1,100 do
      if self.syn_mode then
        self.vrep:stepsForward(10)
      end
      if torch.Tensor(self:getArmVelocity('left')):abs():sum() < 0.001 then
        break
      elseif i==100 then
        print("Failed to set the arm pose!!!")
      end
    end
  end

  self.collision = false
  self.reached = false
  -- rs = self.vrep:setImageRendering(true)

  if self.ctrl_mode ~= 'position' then
    self.vrep:enableJointVelocityCtrl()
  end

  return target, pose, desired_pose
end

-- Get config for a target position
function sim:getTargetConfig(target_pos)
  -- self.vrep:setObjectOrientation(self.cuboid, self.robot_base, {0.0,0.0,0.0})
  self.vrep:setObjectPosition(self.cuboid, self.robot_base, target_pos)
  local desired_pose = self.vrep:getDesiredConfig(0.2)
  -- for i=1,100 do
  --   if desired_pose~=nil then
  --     if #desired_pose~=0 then
  --       -- print("*******",i)
  --       break
  --     end
  --   end
  --   desired_pose = self.vrep:getDesiredConfig(0.2)
  -- end

  -- self.vrep:setObjectOrientation(self.cuboid, self.table_top, {0.0,0.0,0.0})
  self.vrep:setObjectPosition(self.cuboid, self.robot_base, self.target)

  return desired_pose

end

-- Generate a random target
function sim:generateRandTarget()--, joints_range)
  local target = {}
  if self.target_constant then
    target = table.copy(self.target_constant)
  else
    for i=1,#self.target_min do
      target[i] = self.target_min[i] + torch.uniform() * (self.target_max[i] - self.target_min[i])
    end
  end

  return target
end

-- Set target
function sim:setTarget(target)--, joints_range)
  self.vrep:setObjectOrientation(self.cuboid, self.robot_base, {0.0,0.0,0.0})
  self.vrep:setObjectPosition(self.cuboid, self.robot_base, target)
  target = self.vrep:getObjectPosition(self.cuboid, self.robot_base)
  for i=1,100 do
    if target then
      break
    end
    target = self.vrep:getObjectPosition(self.cuboid, self.robot_base)
  end

  return target
end

-- Initialize the arm to an initial pose
function sim:initArmPose()
  local pose = table.copy(self.left_arm_init_pose)
  -- local pose = self.left_arm_init_pose

  self:setArmTargetPose('left',pose)

  -- Wait the arm gets to the desired pose
  if self.syn_mode then
    self.vrep:stepsForward()
  end
  local vel = self:getArmVelocity('left')
  local sum_vel = torch.Tensor(vel):abs():sum()
  while sum_vel > 0.01 do
    if self.syn_mode then
      self.vrep:stepsForward(10)
    end
    vel = self:getArmVelocity('left')
    sum_vel = torch.Tensor(vel):abs():sum()
  end

  return pose
end

-- Generate a random arm pose
function sim:generateRandPose(pose, desired_pose)
  if self.pose_constant then
    return pose
  else
    local rand_pose = {}
    for k=1,10 do
      for i=1,self.n_joint do
        rand_pose[i] = pose[i] + torch.uniform() * (desired_pose[i] - pose[i])
      end

      self:setArmTargetPose('left',rand_pose)
      -- Wait the arm gets to the desired pose
      if self.syn_mode then
        self.vrep:stepsForward()
      end
      local vel = self:getArmVelocity('left')
      local sum_vel = torch.Tensor(vel):abs():sum()
      for k=1,100 do
      -- while sum_vel > 0.01 do
        if self.syn_mode then
          self.vrep:stepsForward(10)
        end
        vel = self:getArmVelocity('left')
        sum_vel = torch.Tensor(vel):abs():sum()
        if sum_vel < 0.01 then
          -- print(k)
          break
        end
      end
      local collision_, reached = self:detectCollisions()
      print("Rand: ", k)
      if not collision_ and not reached then
        break
      end
    end

    return rand_pose
  end
end

function sim:setCameraPose(random)
  local camera_pose = table.copy(self.right_arm_init_pose)
  if random then
    for i=1,self.n_joint do
      camera_pose[i] = camera_pose[i] + (torch.uniform() - 0.5) * self.camera_pose_range
    end
  end

  self:setArmTargetPose('right',camera_pose)
end

function sim:randTargetOrientation()
  local orientation_z = torch.uniform() * 2 * math.pi
  return {0.0,0.0,orientation_z}
end

function sim:randTablePose(random)
  -- Randomly set the table position and orientation
  local table_position = table.copy(self.table_position)
  local table_orientation = table.copy(self.table_orientation)

  if random then
    for i=1,3 do
      table_position[i] = table_position[i] + (torch.uniform() - 0.5) * self.table_position_range[i]
      table_orientation[i] = table_orientation[i] + (torch.uniform() - 0.5) * self.table_orientation_range[i]
    end
  end

  self.vrep:setObjectOrientation(self.table, -1, table_orientation)
  self.vrep:setObjectPosition(self.table, -1, table_position)

end

function sim:randClutters()
  -- Randomly set the tabletop clutters

  -- number of clutter objects
  local objects_display = {}
  local clutters_h = {}
  local objects_remove = {}
  for i=1,self.n_clutters do
    if torch.uniform() < 0.5 and not self.no_clutters then
      objects_display[#objects_display+1] = self.clutters_handles[i]
      clutters_h[#clutters_h+1] = self.clutters_h[i]
    else
      objects_remove[#objects_remove+1] = self.clutters_handles[i]
    end
  end
  self:removeObjects(objects_remove)
  self:randObjects(objects_display, clutters_h)


  -- clutter shapes

  -- clutter colors

  -- clutter pose


end

function sim:removeObjects(clutters)
  local n = #clutters
  for i=1,n do
    self.vrep:setObjectPosition(clutters[i], self.table, {0.0, 0.0, 0.0})
  end
end

function sim:randObjects(clutters, clutters_h)
  local n = #clutters
  for i=1,n do
    -- self.vrep:scaleObjectSize(clutters[i], {0.1*torch.uniform(),0.1*torch.uniform()})
    self.vrep:setObjectPosition(clutters[i], self.table_top, {(torch.uniform()-0.35)*0.9, (torch.uniform()-0.5)*0.6, clutters_h[i]})
    self.vrep:setObjectOrientation(clutters[i], self.table_top, {0.0,0.0,torch.uniform() * 2 * math.pi})
  end
  self.vrep:setObjectOrientation(self.clutters_handles[5], -1, {torch.uniform()*2*math.pi,0.5*math.pi,0.0})
  -- self.vrep:setObjectOrientation(self.clutters_handles[5], -1, {torch.uniform()*2*math.pi,0.0,0.0})
end

function sim:picking(action, new, testing)
  if self.real_world then
    return self:picking_real(action, new, testing)
  else
    if self.RL_mode then
      return self:picking_sim_RL(action, new, testing)
    else
      return self:picking_sim(action, new, testing)
    end
  end
end

-- Simulation main function
function sim:picking_sim_RL(action, new, testing)
  local collision, reached, im
  if new then -- when starting a new game, reset the joints to initial pose and randomly get a new destination
    self.target, self.left_joint_position, self.desired_pose = self:initScene()
    print("Current Target: ", torch.Tensor(self.target))
    print("Current Arm Pose: ", torch.Tensor(self.left_joint_position))
  else
    if self.ctrl_mode == 'position' then
      -- self:ctrlPosition(action)
      self:ctrlPosition_15(action)
    elseif self.ctrl_mode == 'dis_velocity' then
      self:ctrlSpeedDiscrete(action)
    elseif self.ctrl_mode == 'velocity' then
      -- local vel = {}
      -- for i=1,self.n_joint do
      --   vel[i] = self.desired_pose[i] - self.left_joint_position[i]
      --   if vel[i] > 1.0 then
      --     vel[i] = 1.0
      --   elseif vel[i] < -1.0 then
      --     vel[i] = -1.0
      --   end
      -- end
      self:ctrlSpeedContinuous(action)
    end
  end

  collision, reached = self:getMeasurements()
  if self.image_format == 'RGB' then
    im = self.vrep:getRGBImage(self.right_camera)
  else
    im = self.vrep:getGreyImage(self.right_camera)
  end

  if self.image_res[1] == 400 then
    im = image.crop(im,0,0,400,400) -- 400*400
  else
    im = image.crop(im,120,20,420,320) -- 300*300
  end
  im = im:float():div(255)

  -- Generate low dimensional features
  local low_dim_features
  if self.e2e_mode then
    low_dim_features = self:outputLowDimTensor(self.left_joint_position)
  else
    low_dim_features = self:outputLowDimTensor(self.left_joint_position, self.target)
  end
  local configuration = {current_pos=self.left_joint_position, desired_pos=self.desired_pose, action_step=self.action_step}

  if testing then
    local reward, termination, completion, closest_distance, distance = self.rwd_ob:reward1_testing(self.distance_e2t, collision, reached)
    return im, reward, termination, completion, low_dim_features, configuration, closest_distance, distance
  else
    local reward, termination, completion = self.rwd_ob:reward1(self.distance_e2t, collision, reached)
    return im, reward, termination, completion, low_dim_features, configuration
  end

  -- print("distance: ", self.distance_e2t)
  -- print("reward: ", reward)
  -- print("termination: ", termination)
  -- print("completion: ", completion)
  -- print("low dim features: ", low_dim_features)

end


-- Simulation main function
function sim:picking_sim(action, new, testing)
  local termination = false
  local reached = false
  if new then -- when starting a new game, reset the joints to initial pose and randomly get a new destination
    self.target, self.left_joint_position, self.desired_pose = self:initScene()
    print("Current Target: ", torch.Tensor(self.target))
    -- print("Current Arm Pose: ", torch.Tensor(self.left_joint_position))
    self.used_steps = 0
    self.completion = false
    self.closest_distance = 100
  else
    self.used_steps = self.used_steps + 1
    if self.ctrl_mode == 'position' then
      -- self:ctrlPosition(action)
      self:ctrlPosition_15(action)
    elseif self.ctrl_mode == 'dis_velocity' then
      self:ctrlSpeedDiscrete(action)
    elseif self.ctrl_mode == 'velocity' then
      -- local vel = {}
      -- for i=1,self.n_joint do
      --   vel[i] = self.desired_pose[i] - self.left_joint_position[i]
      --   if vel[i] > 1.0 then
      --     vel[i] = 1.0
      --   elseif vel[i] < -1.0 then
      --     vel[i] = -1.0
      --   end
      -- end
      if torch.Tensor(action):abs():sum() < 0.05 then -- 0.05, 0.01
          action = {0.0,0.0,0.0,0.0,0.0,0.0,0.0}
      end
      collision, reached = self:ctrlSpeedContinuous(action)
      if torch.Tensor(action):abs():sum() < 0.05 or self.used_steps > self.max_step then
          termination = true
      end
    end
  end

  -- local collision, reached = self:getMeasurements()
  self:getMeasurements()
  local distance = self.distance_e2t
  if distance < self.closest_distance then
    self.closest_distance = distance
  end
  if reached then
    self.completion = true
  end

  local im
  if self.image_format == 'RGB' then
    im = self.vrep:getRGBImage(self.right_camera)
  else
    im = self.vrep:getGreyImage(self.right_camera)
  end

  if self.image_res[1] == 400 then
    im = image.crop(im,0,0,400,400) -- 400*400
  else
    im = image.crop(im,120,20,420,320) -- 300*300
  end
  im = im:float():div(255)

  -- Generate low dimensional features
  local low_dim_features
  if self.e2e_mode then
    low_dim_features = self:outputLowDimTensor(self.left_joint_position)
  else
    local target = table.copy(self.target)
    if self.upper_plane then
      target[3] = target[3] + self.target_h
    end
    low_dim_features = self:outputLowDimTensor(self.left_joint_position, target)
  end
  local configuration = {current_pos=self.left_joint_position, desired_pos=self.desired_pose, action_step=self.action_step}


  local step_cost = 1
  return im, step_cost, termination, self.completion, low_dim_features, configuration, self.closest_distance, distance

  -- print("distance: ", self.distance_e2t)
  -- print("reward: ", reward)
  -- print("termination: ", termination)
  -- print("completion: ", completion)
  -- print("low dim features: ", low_dim_features)

end

-- Evaluation at the end of each reached
function sim:eval_accuracy(current_pos, target)
    if target == nil then
        print("<<<<<<<<<<<<<<<<<<<< Please put the end-effector of the left arm to the center of the object upper plane!")
        self.inter.waitKeyEnter()
        self.left_joint_position = self.inter.getArmPose()
        target = table.copy(self.left_joint_position)
    end

    self:setArmTargetPose('left',current_pos)
    for i=1,100 do
      if torch.Tensor(self:getArmVelocity('left')):abs():sum() < 0.001 then
        break
      elseif i==100 then
        print("Failed to get the end-effector pose for current arm pose!!!")
      end
    end
    local position_e = self.vrep:getObjectPosition(self.left_gripper, self.robot_base)
    -- position_e[3] = position_e[3] - 0.0325

    -- local position_tttt = self.vrep:getObjectPosition(self.cuboid, self.robot_base)
    -- print(torch.Tensor(position_tttt))

    target = torch.Tensor(target)
    position_e = torch.Tensor(position_e)

    local dis_e = target:csub(position_e)
    local e_2 = torch.pow(dis_e,2)
    local e = torch.sum(e_2)
    e = math.sqrt(e)

    return e

end

-- Real picking main function
function sim:picking_real(action, new, testing)
  local collision, reached, im
  local termination, completion
  local closest_distance = 0.01
  local distance = 0.01
  local reward = 1


  if new then -- when starting a new game, reset the joints to initial pose and randomly get a new destination
    self.target, self.left_joint_position = self:initScene()
    self.target[3] = self.target[3] + 0.0325
    self.used_steps = 0
    print("Current Target: ", torch.Tensor(self.target))
    print("Current Arm Pose: ", torch.Tensor(self.left_joint_position))
  else
    self.used_steps = self.used_steps + 1
    if self.ctrl_mode == 'position' then
      -- self:ctrlPosition(action)
      self:ctrlPosition_15(action)
    elseif self.ctrl_mode == 'dis_velocity' then
      self:ctrlSpeedDiscrete(action)
    elseif self.ctrl_mode == 'velocity' then
        -- Added for testing with conventional controller
        if action == nil then
            self:setArmTargetPose('left',self.left_joint_position)
            local desired_pose = self.vrep:getDesiredConfig(0.2)
            for i=1,100 do
                if desired_pose ~= nil then
                    if #desired_pose ~= 0 then
                        local vel = {}
                        for ii=1,self.n_joint do
                          vel[ii] = desired_pose[ii] - self.left_joint_position[ii]
                          if vel[ii] > 1.0 then
                            vel[ii] = 1.0
                          elseif vel[ii] < -1.0 then
                            vel[ii] = -1.0
                          end
                        end
                        local vel_sum = torch.Tensor(vel):abs():sum()
                        -- print("vel sum: ", vel_sum)
                        -- if vel_sum < 0.01 then
                        if vel_sum < 0.001 then
                          vel = {0.0,0.0,0.0,0.0,0.0,0.0,0.0}
                        --   termination = true
                          -- completion = true
                        end
                        action = vel
                        break
                    end
                end
                desired_pose = self.vrep:getDesiredConfig(0.2)
            end

        -- Added for testing with conventional controller plus a perception module
        elseif  #action == 3 then
            print("predicted target position: ",torch.Tensor(action))
            self.vrep:setObjectPosition(self.cuboid, self.robot_base, action)
            self:setArmTargetPose('left',self.left_joint_position)
            local desired_pose = self.vrep:getDesiredConfig(0.2)
            for i=1,100 do
                if desired_pose ~= nil then
                    if #desired_pose ~= 0 then
                        local vel = {}
                        for ii=1,self.n_joint do
                          vel[ii] = desired_pose[ii] - self.left_joint_position[ii]
                          if vel[ii] > 1.0 then
                            vel[ii] = 1.0
                          elseif vel[ii] < -1.0 then
                            vel[ii] = -1.0
                          end
                        end
                        local vel_sum = torch.Tensor(vel):abs():sum()
                        -- print("vel sum: ", vel_sum)
                        -- if vel_sum < 0.01 then
                        if vel_sum < 0.001 then
                          vel = {0.0,0.0,0.0,0.0,0.0,0.0,0.0}
                        --   termination = true
                          -- completion = true
                        end
                        action = vel
                        break
                    end
                end
                desired_pose = self.vrep:getDesiredConfig(0.2)
            end

        end
        if torch.Tensor(action):abs():sum() < 0.05 then -- 0.05, 0.01
          action = {0.0,0.0,0.0,0.0,0.0,0.0,0.0}
        end
        self:ctrlSpeedContinuous(action)
        if torch.Tensor(action):abs():sum() < 0.05 or self.used_steps > self.max_step then
          if self.sim_online then
              distance = self:eval_accuracy(self.left_joint_position, self.target)
          end
          termination = true
          completion = true
        --   print(">>>>>>>>>>>>>>>>>>>> Please measure the final distance !!!!!!!!!!!!!!!!!!!!!")
        --   self.inter.waitKeyEnter()
        end
    end
  end

  -- if self.image_format == 'RGB' then
  --   im = self.vrep:getRGBImage(self.right_camera)
  -- else
  --   im = self.vrep:getGreyImage(self.right_camera)
  -- end
  -- im = py.eval(self.inter.getRGBImage())
  im = self.inter.getRGBImage()

  im = self:stringI2torchTensor(im, {400,640})
  if self.image_res[1] == 400 then
    im = image.crop(im,0,0,400,400) -- 400*400
  else
    im = image.crop(im,120,20,420,320) -- 300*300
  end

  -- im = image.crop(im,100,20,400,320)
  im = im:float():div(255)
  -- x:set(s, 1, sz)
  -- print(im:type())

  -- Generate low dimensional features
  local low_dim_features
  if self.e2e_mode then
    low_dim_features = self:outputLowDimTensor(self.left_joint_position)
  else
    low_dim_features = self:outputLowDimTensor(self.left_joint_position, self.target)
  end
  -- local low_dim_features = self:outputLowDimTensor(self.left_joint_position)
  -- local configuration = {current_pos=self.left_joint_position, desired_pos=self.desired_pose, action_step=self.action_step}

  if testing then
    -- local reward, termination, completion, closest_distance, distance = self.rwd_ob:reward1_testing(self.distance_e2t, collision, reached)
    return im, reward, termination, completion, low_dim_features, configuration, closest_distance, distance
  else
    -- local reward, termination, completion = self.rwd_ob:reward1(self.distance_e2t, collision, reached)
    return im, reward, termination, completion, low_dim_features, configuration
  end

  -- print("distance: ", self.distance_e2t)
  -- print("reward: ", reward)
  -- print("termination: ", termination)
  -- print("completion: ", completion)
  -- print("low dim features: ", low_dim_features)

end

-- Convert a string image received from VREP to a tensor
function sim:stringI2torchTensor(i, res)
  local n_pixel = res[1] * res[2]
  local s_len = string.len(i)

  -- Convert a string image to a tensor
  local im = torch.ByteTensor(s_len)
  for k=1,s_len do
    im[k] = string.byte(i,k)
  end

  if s_len == 3 * n_pixel then -- RGB image
    im = im:resize(3,res[1],res[2])
    -- local im_temp = torch.ByteTensor(3,res[1],res[2])
    -- for j=1,3 do
    --   im_temp:select(1,j):copy(im:select(1,4-j))
    -- end
    -- im = im_temp
  else -- Grey-scale image
    im = im:resize(res[1],res[2])
  end

  -- im = image.vflip(im)

  return im

end


--[[ Function
@description: add random noises on an image tensor (uniform noises)
@input:
    image_: an image tensor where the noises will be added
    scale: the value range of the noises, i.e., 0.2 (-0.1~0.1)
@output:
    image_: the image tensor added with noises
@notes:
]]
function sim:addRandomNoises(image_, scale)
    image_ = torch.rand(image_:size()):add(-0.5):mul(scale):add(image_)
    return image_
end

-- Normalize low dimensional features
function sim:outputLowDimTensor(arm_pose, destin)
    local state = {}

    if arm_pose then
        -- Integrate arm pose features
        for i=1, self.n_joint do
            state[i] = (arm_pose[i] - self.joint_pose_min[i]) / (self.joint_pose_max[i] - self.joint_pose_min[i])
        end
    end

    -- Integrate destination pose features
    if destin then
        local destin_dim = #destin
        local pos_dim = #state
        for i=1, destin_dim do
            local diff = self.target_pose_max[i] - self.target_pose_min[i]
            if diff ~= 0 then
              state[pos_dim+i] = (destin[i] - self.target_pose_min[i]) / diff
            else
              state[pos_dim+i] = destin[i]
            end
        end
    end

    local state_ = torch.Tensor(state)
    -- print(state_)

    return state_
end

-- Recover normalized low-dim features to their original values
function sim:recoverLowDimTensor(state)
    local state = state:clone()

    local n = state:size(1)
    -- Recover destination pose features
    for i=1, n do
        state[i] = state[i] * (self.target_pose_max[i] - self.target_pose_min[i]) + self.target_pose_min[i]
    end

    return state
end


function sim:get_one_frame()
    im = self.inter.getRGBImage()
    im = self:stringI2torchTensor(im, {400,640})
    if self.image_res[1] == 400 then
      im = image.crop(im,0,0,400,400) -- 400*400
    else
      -- im = image.crop(im,100,20,400,320)
      im = image.crop(im,120,20,420,320) -- 300*300
    end

    im = im:float():div(255)

    return im
end



-- Generate a dataset for perception
function sim:collect_real_dataset()
  -- TODO: to also save the raw images

  local n_targets = 50
  -- local n_targets = 30
  local images_for_each = 3
  local image400 = {}
  local image300 = {}
  local target_arm_pose = {}
  local arm_pose = {}
  local target_pose
  local sample_index = 0

  for i=1,n_targets do
    print("<<<<<<<<<<<<<<<<<<<<1.Pose for target position " .. i .. "!")
    self.inter.waitKeyEnter()
    self.left_joint_position = self.inter.getArmPose()
    target_pose = table.copy(self.left_joint_position)
    -- Record Arm pose
    for j=1,images_for_each do
      self.inter.initHead(0.53)
      self.inter.initCam(self.right_arm_init_pose)
      print(">>>>>>>>>>>>>>>>>>>>2.Random Arm Pose " .. j .. "!")
      self.inter.waitKeyEnter()
      self.left_joint_position = self.inter.getArmPose()
      -- "Randomize Camera"

      local im = self.inter.getRGBImage()
      im = self:stringI2torchTensor(im, {400,640})
      local im400 = image.crop(im,0,0,400,400) -- 400*400
      local im300 = image.crop(im,120,20,420,320) -- 300*300
      im400 = im400:float():div(255)
      im300 = im300:float():div(255)

      -- Record Data for each sample
      sample_index = sample_index + 1
      image400[sample_index] = im400
      image300[sample_index] = im300
      target_arm_pose[sample_index] = table.copy(target_pose)
      arm_pose[sample_index] = table.copy(self.left_joint_position)

      print("=====================================")
      print("Sample Index: ", sample_index)
      print("Traget Arm Pose: ", torch.Tensor(target_arm_pose[sample_index]))
      print("Current Arm Pose: ", torch.Tensor(arm_pose[sample_index]))
      win_input1 = image.display{image=image400[sample_index], win=win_input1}
      win_input2 = image.display{image=image300[sample_index], win=win_input2}
      -- print("Label V: ", torch.Tensor(label_v_[#label_v_]))

      -- Save the dataset to a t7 file
      local filename_ = "image300_real_data_"
      filename_ = filename_ .. n_targets .. "_" .. images_for_each
      torch.save(filename_ .. ".t7", {image = image300,
                              target_arm_pose = target_arm_pose,
                              arm_pose = arm_pose,
                              sample_amount = sample_index})
      print('Real dataset saved:', filename_ .. '.t7')
      filename_ = "image400_real_data_"
      filename_ = filename_ .. n_targets .. "_" .. images_for_each
      torch.save(filename_ .. ".t7", {image = image400,
                              target_arm_pose = target_arm_pose,
                              arm_pose = arm_pose,
                              sample_amount = sample_index})
      print('Real dataset saved:', filename_ .. '.t7')

    end
  end
  io.flush()
  collectgarbage()

  print("Data Collection Done!!! Go for a break!!!")

end


function table.copy(t)
    if t == nil then return nil end
    local nt = {}
    for k, v in pairs(t) do
        if type(v) == 'table' then
            nt[k] = table.copy(v)
        else
            nt[k] = v
        end
    end
    setmetatable(nt, table.copy(getmetatable(t)))
    return nt
end

function sim:load_pre_constructed_dataset(file)
    -- try to load the dataset
    local err_msg,  dataset= pcall(torch.load, file)
    if not err_msg then
        error("Could not find the dataset file ")
    end
    return dataset
end
