--[[ File
@description:
    This class is for generating different datasets.
@version: V0.02
@author: Fangyi Zhang   email:gzzhangfangyi@gmail.com
@acknowledgement:
    ARC Centre of Excellence for Robotic Vision (ACRV)
    Queensland Univsersity of Technology (QUT)
@history:
    V0.00   07/06/2017  developed the first version
    V0.01   11/06/2017  fixed the bug of not deeply copying the table in the dataset
    V0.02   19/06/2017  made compatible to end-to-end reaching with a complex network
    V0.03   14/05/2019  removed the reliance on interface_vrep in offline mode
]]


require 'torch'
-- require 'common/interface_vrep'
require 'image'
-- local interface_vrep = require 'common/interface_vrep'
-- local reward_functions_picking = require 'reward/reward_functions_picking'


-- construct a class
local dctrl = torch.class('dataset_ctrl')


--[[ Function
@description: initialize an object for manipulation simulation
@input:
    args: settings for a manipulation simulation object
@output: nil
@notes:
]]
function dctrl:__init(args)
    -- vrep setup
    self.syn_mode = args.syn_mode or false
    self.sim_step = args.sim_step or 50 -- 50 ms
    self.image_format = args.image_format or 'Grey'
    self.image_res = args.image_res or {300,300}
    self.online_mode = args.online_mode or false
    self.max_step = args.max_step or 400

    self.left_joint_position = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}
    self.right_joint_position = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}
    self.left_joint_velocity = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}
    self.right_joint_velocity = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}

    -- Settings for Clutters
    self.clutters = args.clutters or {'Baseball','Block','Crayons','Dove','Eraser','EraserPack','Note1','RubCube','UQBall'}
    self.clutters_h = args.clutters_h or {0.0375,0.01,0.0125,0.0175,0.03,0.0175,0.005,0.029,0.03}
    self.n_clutters = #self.clutters

    -- Initialize vrep
    if self.online_mode then
      require 'common/interface_vrep'
      self:initVREP(args)
    else
      if args.dataset then
        -- Load pre-generated datasets
        self.dataset = self:load_pre_constructed_dataset(args.dataset)
        -- Decide whether the dataset satisfies current accuracy requirements
        if self.image_res[1] == 400 then
          self.dataset_image = self.dataset.image400
        else
          self.dataset_image = self.dataset.image300
          -- self.dataset_image = self.dataset.image
          -- if self.dataset_image == nil then
          --   self.dataset_image = self.dataset.image300
          -- end
        end
        if self.dataset_image == nil then
          self.dataset_image = self.dataset.image
        end
        self.dataset_label = self.dataset.label
        if self.dataset_label == nil then
          self.dataset_label = self.dataset.object_position
        end
        self.dataset_s = self.dataset.low_dim_s_e
        if self.dataset_s == nil then
          self.dataset_s = self.dataset.low_dim_s
        end
        self.dataset_vel = self.dataset.vel
        self.dataset_position = self.dataset.object_position
        self.dataset_arm_pose = self.dataset.arm_pose
        self.dataset_L = self.dataset.distance_e2t
        self.dataset_sample_amount = self.dataset.sample_amount
        io.flush()
        collectgarbage()
      end
    end

    if args.extra_dataset then
      -- Load pre-generated datasets
      self.extra_dataset = self:load_pre_constructed_dataset(args.extra_dataset)
      -- Decide whether the dataset satisfies current accuracy requirements
      if self.image_res[1] == 400 then
        self.extra_dataset_image = self.extra_dataset.image400
      else
        self.extra_dataset_image = self.extra_dataset.image
        if self.extra_dataset_image == nil then
          self.extra_dataset_image = self.extra_dataset.image300
        end
      end
      self.extra_dataset_label = self.extra_dataset.label
      if self.extra_dataset_label == nil then
        self.extra_dataset_label = self.extra_dataset.object_position
      end
      self.extra_dataset_sample_amount = self.extra_dataset.sample_amount
      io.flush()
      collectgarbage()
    end

    if args.real_dataset then
      -- Load pre-generated datasets
      self.real_dataset = self:load_pre_constructed_dataset(args.real_dataset)
      -- Decide whether the dataset satisfies current accuracy requirements
      self.real_dataset_image = self.real_dataset.image
      self.real_dataset_label = self.real_dataset.object_position
      self.real_dataset_vel = self.real_dataset.vel
      self.real_dataset_arm_pose = self.real_dataset.arm_pose
      self.real_dataset_s = self.real_dataset.low_dim_s_e
      self.real_dataset_sample_amount = self.real_dataset.sample_amount
      io.flush()
      collectgarbage()
    end

    if args.e2e_dataset then
      -- Load pre-generated datasets
      self.e2e_dataset = self:load_pre_constructed_dataset(args.e2e_dataset)
      -- Decide whether the dataset satisfies current accuracy requirements
      self.e2e_dataset_image = self.e2e_dataset.image300
      self.e2e_dataset_label = self.e2e_dataset.object_position
      self.e2e_dataset_vel = self.e2e_dataset.vel
      self.e2e_dataset_arm_pose = self.e2e_dataset.arm_pose
      self.e2e_dataset_s = self.e2e_dataset.low_dim_s_e
      self.e2e_dataset_sample_amount = self.e2e_dataset.sample_amount
      io.flush()
      collectgarbage()
    end

    if args.e2e_real_dataset then
      -- Load pre-generated datasets
      self.e2e_real_dataset = self:load_pre_constructed_dataset(args.e2e_real_dataset)
      -- Decide whether the dataset satisfies current accuracy requirements
      self.e2e_real_dataset_image = self.e2e_real_dataset.image
      self.e2e_real_dataset_label = self.e2e_real_dataset.object_position
      self.e2e_real_dataset_vel = self.e2e_real_dataset.vel
      self.e2e_real_dataset_arm_pose = self.e2e_real_dataset.arm_pose
      self.e2e_real_dataset_s = self.e2e_real_dataset.low_dim_s_e
      self.e2e_real_dataset_sample_amount = self.e2e_real_dataset.sample_amount
      io.flush()
      collectgarbage()
    end

    if args.test_dataset then
      -- Load pre-generated datasets
      self.test_dataset = self:load_pre_constructed_dataset(args.test_dataset)
      -- Decide whether the dataset satisfies current accuracy requirements
      self.test_dataset_image = self.test_dataset.image
      self.test_dataset_label = self.test_dataset.object_position
      self.test_dataset_vel = self.test_dataset.vel
      self.test_dataset_arm_pose = self.test_dataset.arm_pose
      self.test_dataset_s = self.test_dataset.low_dim_s_e
      self.test_dataset_sample_amount = self.test_dataset.sample_amount
      io.flush()
      collectgarbage()
    end

    -- Task parameters
    self.weighted_losses = args.weighted_losses or false
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
    -- self.target, self.left_joint_position = self:initScene()

    self.joint_pose_max = args.joint_pose_max or {1.702, 1.048, 3.055, 2.619, 3.060, 2.095, 3.060}
    self.joint_pose_min = args.joint_pose_min or {-1.702, -2.148, -3.055, -0.06, -3.060, -1.571, -3.060}
    self.target_pose_max = args.target_pose_max or {1.019, 0.6617, 0.4584}
    self.target_pose_min = args.target_pose_min or {0.419, 0.1367, 0.3084}
    self.target_max = args.target_max or {1.019, 0.6617, 0.4584}
    self.target_min = args.target_min or {0.419, 0.1367, 0.3084}
    self.camera_pose_range = args.camera_pose_range or 0.02
    self.table_position = args.table_position or  {0.635, 0.15, 0.75}
    self.table_position_range = args.table_position_range or {0.03, 0.1, 0.02}
    self.table_orientation = args.table_orientation or {0.0, 0.0, 1.57}
    self.table_orientation_range = args.table_orientation_range or {0.0, 0.0, 0.14}

    -- Training Settings
    self.add_noise = args.add_noise or false
    self.noisy_prob = args.noisy_prob or 1
    self.upper_plane = args.upper_plane or false
    self.target_h = args.target_h or 0.0325
    if self.upper_plane then
      self.target_pose_min[3] = self.target_pose_min[3] + self.target_h
      self.target_pose_max[3] = self.target_pose_max[3] + self.target_h
    end

    -- Print parameters
    self:report()

    -- self:setArmTargetPose('right',{-0.3,-1.3,1,1.1,0.4,1.8,-1.57})

end

-- Print parameters
function dctrl:report()
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
function dctrl:initVREP(args)
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
function dctrl:getObjectHandles()

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
function dctrl:getCollisionHandles()
  self.collision_arm_handle = self.vrep:getCollisionHandle('Collision_leftArm')
  self.collision_cuboid_handle = self.vrep:getCollisionHandle('Collision_cuboid')
end

-- Get distance handles
function dctrl:getDistanceHandles()
  self.distance_e2t_handle = self.vrep:getDistanceHandle('Distance_e2t')
end

-- Init streaming signals
function dctrl:initStreams()
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
function dctrl:getArmPose(arm)
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
function dctrl:setArmTargetPose(arm, pos)
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
function dctrl:getArmVelocity(arm)
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
function dctrl:setArmVelocity(arm, vel)
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
function dctrl:getMeasurements()
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
function dctrl:detectCollisions()
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
function dctrl:oneControlStep()
  -- detection
  -- local collision = false
  -- local reached =false
  -- local collision_
  if self.syn_mode then
    for i=1,self.n_sim_one_ctrl do
      self.vrep:stepsForward(1)
      -- print("Forard 1 step!!!!")
      -- if detect_collision then
      --   collision_, reached = self:detectCollisions()
      --   if collision_ and not reached then
      --     collision = true
      --   end
      -- end
      -- print("Simulation Time: ", self.vrep:getSimTime())
    end
  end

  -- return collision, reached

end

-- Speed control mode
function dctrl:ctrlSpeedContinuous(vel)
  self:setArmVelocity('left', vel)
  self:oneControlStep()
end

-- Initialize a random scene
function dctrl:initScene()
  -- local rs = self.vrep:setImageRendering(false)
  if self.ctrl_mode ~= 'position' then
    self.vrep:enableJointPositionCtrl()
  end
  local pose = self:initArmPose()
  local target = self:generateRandTarget()
  local desired_pose = self.vrep:getDesiredConfig(0.2)
  pose = self:generateRandPose(pose, desired_pose)
  for i=1,100 do
    if desired_pose~=nil then
      if #desired_pose~=0 then
        -- print("*******",i)
        break
      end
    end
    target = self:generateRandTarget()
    desired_pose = self.vrep:getDesiredConfig(0.2)
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
function dctrl:getTargetConfig(target_pos)
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
function dctrl:generateRandTarget()--, joints_range)
  local target
  if self.target_constant then
    target = table.copy(self.target_constant)
  else
    self.target_x = self.safe_edge + torch.rand(1)[1] * (self.target_region_x  - 2 * self.safe_edge)
    self.target_y = self.safe_edge + torch.rand(1)[1] * (self.target_region_y  - 2 * self.safe_edge) - 0.5 * self.target_region_y
    target = {self.target_x, self.target_y, self.target_z}
  end

  self.vrep:setObjectOrientation(self.cuboid, self.table_top, {0.0,0.0,0.0})
  self.vrep:setObjectPosition(self.cuboid, self.table_top, target)
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
function dctrl:initArmPose()
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
function dctrl:generateRandPose(pose, desired_pose)
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

--[[ Function
@description: add random noises on an image tensor (uniform noises)
@input:
    image_: an image tensor where the noises will be added
    scale: the value range of the noises, i.e., 0.2 (-0.1~0.1)
@output:
    image_: the image tensor added with noises
@notes:
]]
function dctrl:addRandomNoises(image_, scale)
    image_ = torch.rand(image_:size()):add(-0.5):mul(scale):add(image_)
    return image_
end

-- Normalize low dimensional features
function dctrl:outputLowDimTensor(arm_pose, destin)
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
function dctrl:recoverLowDimTensor(state)
    local state = state:clone()

    local n = state:size(1)
    -- Recover destination pose features
    for i=1, n do
        state[i] = state[i] * (self.target_pose_max[i] - self.target_pose_min[i]) + self.target_pose_min[i]
    end

    return state
end

function dctrl:varying_white_balance(im)

  local im_r = im:select(1, 1)
  local im_g = im:select(1, 2)
  local im_b = im:select(1, 3)
  local var_scalar = 0.05

  im_r:mul(1 + 2 * var_scalar * (torch.uniform() - 0.5))
  im_g:mul(1 + 2 * var_scalar * (torch.uniform() - 0.5))
  im_b:mul(1 + 2 * var_scalar * (torch.uniform() - 0.5))

  im = torch.clamp(im,0,1)

  return im

end

-- Normalize images to the ranges [-1,1]
function dctrl:image_normalization(im_)
    -- -- Nor1
    -- local im_nored = image.minmax{tensor=im_}
    --
    -- -- Nor2
    -- im_nored:mul(2):add(-1)
    --
    -- return im_nored

    -- Nor3
    -- im_:mul(2):add(-1)


    return im_
end

-- Get one sample from a dataset
function dctrl:get_one_sample_L()
    -- Randomly select a sample from the dataset
    local index = 1 + math.floor(self.dataset_sample_amount * torch.rand(1)[1])
    if index > self.dataset_sample_amount then
      index = self.dataset_sample_amount
    end

    local image_ = self.dataset_image[index]:clone()
    local arm_pose_ = table.copy(self.dataset_arm_pose[index])
    local arm_pose_nml = self:outputLowDimTensor(arm_pose_,nil)
    local L = self.dataset_L[index]
    -- local arm_pose = table.copy(self.dataset_arm_pose[index])

    -- local label = self.dataset.label[index]:clone()
    -- local end_effector = {self.dataset.end_effector[index][1], self.dataset.end_effector[index][2]}

    -- Add some noisy factors
    if self.add_noise then
      if torch.uniform() < self.noisy_prob then
        image_ = self:varying_white_balance(image_)
      end
      -- Brightness
      if torch.uniform() < self.noisy_prob then
        --Change brightness by changing the v in hsv sapce
        local im_hsv = image.rgb2hsv(image_)
        local im_v = im_hsv:select(1, 3)
        im_v:mul(1+1.6*(torch.uniform()-0.5))
        image_ = image.hsv2rgb(im_hsv)
        image_ = torch.clamp(image_,0,1)
      end
      -- -- Rotation
      -- if torch.rand(1)[1] < self.noisy_prob then
      --     image_ = image.rotate(image_, 0.2*(torch.rand(1)[1]-0.5))
      --     noise_added = true
      -- end
      -- -- Offset
      -- if torch.rand(1)[1] < self.noisy_prob then
      --     image_ = image.translate(image_, 10*(torch.rand(1)[1]-0.5), 10*(torch.rand(1)[1]-0.5))
      --     noise_added = true
      -- end
      -- -- White noise
      -- if torch.rand(1)[1] < self.noisy_prob then
      --     image_ = self:addRandomNoises(image_, 0.01)
      --     noise_added = true
      -- end
    end

    image_ = self:image_normalization(image_)
    -- if self.upper_plane then
    --   -- print("Centre: ", torch.Tensor(label))
    --   position[3] = position[3] + self.target_h
    --   -- print("Upper Plane: ", torch.Tensor(label))
    -- end

    return image_, arm_pose_nml, L
end

function dctrl:get_batch_L(size)
  local s_set = torch.Tensor(size, 3, unpack(self.image_res))
  local s_q_set = torch.Tensor(size, 7)
  local l_set = torch.Tensor(size, 1)

  -- local sim_size = size
  -- if self.real_dataset and not self.weighted_losses then
  --     sim_size = math.floor(0.125 * size)
  --   --   sim_size = 0
  -- end

  for i=1,size do
    -- Get the sample corresponding to current arm pose
    local image_, arm_pose_nml, loss = self:get_one_sample_L()
    -- if i > sim_size then
    --   image_, position, low_dim_s, vel = self:get_one_sample_e2e_real()
    -- else
      -- image_, position, low_dim_s, vel = self:get_one_sample_e2e()
    -- end

    s_set[i] = image_
    s_q_set[i] = arm_pose_nml
    l_set[i] = torch.Tensor({loss})
  end

  return {s_q_set, s_set}, l_set
end

-- Get one sample from a dataset
function dctrl:get_one_sample()
    -- Randomly select a sample from the dataset
    local index = 1 + math.floor(self.dataset_sample_amount * torch.rand(1)[1])
    if index > self.dataset_sample_amount then
      index = self.dataset_sample_amount
    end

    local image_ = self.dataset_image[index]:clone()
    local label = table.copy(self.dataset_label[index])

    -- local label = self.dataset.label[index]:clone()
    -- local end_effector = {self.dataset.end_effector[index][1], self.dataset.end_effector[index][2]}

    -- Add some noisy factors
    if self.add_noise then
      -- Brightness
      if torch.uniform() < self.noisy_prob then
        image_ = self:varying_white_balance(image_)
      end
      if torch.uniform() < self.noisy_prob then
        -- -- Change brightness by scaling RGB
        -- image_ = image_:mul(1+0.8*(torch.rand(1)[1]-0.95))

        --Change brightness by changing the v in hsv sapce
        local im_hsv = image.rgb2hsv(image_)
        local im_v = im_hsv:select(1, 3)
        im_v:mul(1+1.6*(torch.uniform()-0.5))
        image_ = image.hsv2rgb(im_hsv)
        image_ = torch.clamp(image_,0,1)
      end
      -- -- Rotation
      -- if torch.rand(1)[1] < self.noisy_prob then
      --     image_ = image.rotate(image_, 0.2*(torch.rand(1)[1]-0.5))
      --     noise_added = true
      -- end
      -- -- Offset
      -- if torch.rand(1)[1] < self.noisy_prob then
      --     image_ = image.translate(image_, 10*(torch.rand(1)[1]-0.5), 10*(torch.rand(1)[1]-0.5))
      --     noise_added = true
      -- end
      -- -- White noise
      -- if torch.rand(1)[1] < self.noisy_prob then
      --     image_ = self:addRandomNoises(image_, 0.01)
      --     noise_added = true
      -- end
    end

    image_ = self:image_normalization(image_)
    if self.upper_plane then
      -- print("Centre: ", torch.Tensor(label))
      label[3] = label[3] + self.target_h
      -- print("Upper Plane: ", torch.Tensor(label))
    end

    return image_, label
end


-- Get one sample from a dataset
function dctrl:get_one_test_sample()
    -- Randomly select a sample from the dataset
    local index = 1 + math.floor(self.test_dataset_sample_amount * torch.rand(1)[1])
    if index > self.test_dataset_sample_amount then
      index = self.test_dataset_sample_amount
    end

    local image_ = self.test_dataset_image[index]:clone()
    local label = table.copy(self.test_dataset_label[index])

    -- local label = self.dataset.label[index]:clone()
    -- local end_effector = {self.dataset.end_effector[index][1], self.dataset.end_effector[index][2]}

    -- Add some noisy factors
    if self.add_noise then
      -- Brightness
      if torch.uniform() < self.noisy_prob then
        image_ = self:varying_white_balance(image_)
      end
      if torch.uniform() < self.noisy_prob then
        --Change brightness by changing the v in hsv sapce
        local im_hsv = image.rgb2hsv(image_)
        local im_v = im_hsv:select(1, 3)
        im_v:mul(1+1.6*(torch.uniform()-0.5))
        image_ = image.hsv2rgb(im_hsv)
        image_ = torch.clamp(image_,0,1)
      end
      -- -- Rotation
      -- if torch.rand(1)[1] < self.noisy_prob then
      --     image_ = image.rotate(image_, 0.2*(torch.rand(1)[1]-0.5))
      --     noise_added = true
      -- end
      -- -- Offset
      -- if torch.rand(1)[1] < self.noisy_prob then
      --     image_ = image.translate(image_, 10*(torch.rand(1)[1]-0.5), 10*(torch.rand(1)[1]-0.5))
      --     noise_added = true
      -- end
      -- -- White noise
      -- if torch.rand(1)[1] < self.noisy_prob then
      --     image_ = self:addRandomNoises(image_, 0.01)
      --     noise_added = true
      -- end
    end

    image_ = self:image_normalization(image_)
    if self.upper_plane then
      -- print("Centre: ", torch.Tensor(label))
      label[3] = label[3] + self.target_h
      -- print("Upper Plane: ", torch.Tensor(label))
    end

    return image_, label
end


-- Get a batch with a certain size from a dataset
function dctrl:get_batch(size)
    -- TODO: make the initialization of s_set and l_set autonomous
    local s_set = torch.Tensor(size, 3, unpack(self.image_res))
    local l_set = torch.Tensor(size, 3)
    local sim_size = size
    if self.real_dataset then
        -- sim_size = math.floor(0.5 * size)
        -- sim_size = math.floor(0.25 * size)
        sim_size = 0
    end

    for i=1,size do
      -- Get the sample corresponding to current arm pose
      local image_, label
      if i > sim_size then
        image_, label = self:get_one_sample_real()
      else
        image_, label = self:get_one_sample()
      end
      -- Generate low dimensional features
      local low_dim_features = self:outputLowDimTensor(nil, label)

      s_set[i] = image_
      l_set[i] = low_dim_features
    end

    return s_set, l_set
end


-- Get a batch with a certain size from a dataset
-- For the case of using an auto-encoder for unsupervised domain adaptation
function dctrl:get_batch_auto(size)
    -- TODO: make the initialization of s_set and l_set autonomous
    local sim_size = size
    if self.real_dataset then
        -- sim_size = math.floor(0.5 * size)
        sim_size = math.floor(0.25 * size)
        -- sim_size = 0
    end
    local s_auto_set = torch.Tensor(size, 3, unpack(self.image_res))
    local s_set = torch.Tensor(sim_size, 3, unpack(self.image_res))
    local l_set = torch.Tensor(sim_size, 3)

    for i=1,size do
      -- Get the sample corresponding to current arm pose
      local image_, label
      if i > sim_size then
        image_, label = self:get_one_sample_real()
      else
        image_, label = self:get_one_sample()
        -- Generate low dimensional features
        local low_dim_features = self:outputLowDimTensor(nil, label)
        s_set[i] = image_
        l_set[i] = low_dim_features
      end
      s_auto_set[i] = image_:clone()
    end

    return s_auto_set, s_set, l_set
end


-- Get a batch with a certain size from a dataset
-- For the case of using an auto-encoder for unsupervised domain adaptation
function dctrl:get_batch_adda(size)
    -- TODO: make the initialization of s_set and l_set autonomous
    local s_sim_set = torch.Tensor(size, 3, unpack(self.image_res))
    local s_real_set = torch.Tensor(size, 3, unpack(self.image_res))
    local l_pose_set = torch.Tensor(size, 3)

    for i=1,size do
      -- Get the sample corresponding to current arm pose
      local real_im, label = self:get_one_sample_real()
      local sim_im, _ = self:get_one_sample()
      -- Generate low dimensional features
      local low_dim_features = self:outputLowDimTensor(nil, label)
      s_sim_set[i] = sim_im
      s_real_set[i] = real_im
      l_pose_set[i] = low_dim_features
    end

    return s_sim_set, s_real_set, l_pose_set
end


-- Get a batch with a certain size from a dataset
-- For the case of using domain confusion for unsupervised domain adaptation
function dctrl:get_batch_dc(size)
    -- TODO: make the initialization of s_set and l_set autonomous
    local s_sim_set = torch.Tensor(size, 3, unpack(self.image_res))
    local s_real_set = torch.Tensor(size, 3, unpack(self.image_res))
    local sim_pose_set = torch.Tensor(size, 3)
    local real_pose_set = torch.Tensor(size, 3)

    for i=1,size do
      -- Get the sample corresponding to current arm pose
      local real_im, pose_real = self:get_one_sample_real()
      local sim_im, pose_sim = self:get_one_sample()
      -- Generate low dimensional features
      local low_dim_features_real = self:outputLowDimTensor(nil, pose_real)
      local low_dim_features_sim = self:outputLowDimTensor(nil, pose_sim)
      s_sim_set[i] = sim_im
      s_real_set[i] = real_im
      sim_pose_set[i] = low_dim_features_sim
      real_pose_set[i] = low_dim_features_real
    end

    return s_sim_set, s_real_set, sim_pose_set, real_pose_set
end


-- Get a batch with a certain size from a dataset
-- For the case of using domain confusion for unsupervised domain adaptation
function dctrl:get_batch_dc_semi(size)
    -- TODO: make the initialization of s_set and l_set autonomous
    local s_sim_set = torch.Tensor(size, 3, unpack(self.image_res))
    local s_real_set_t = torch.Tensor(size, 3, unpack(self.image_res))
    local s_real_set = torch.Tensor(size, 3, unpack(self.image_res))
    local sim_pose_set = torch.Tensor(size, 3)
    local real_pose_set_t = torch.Tensor(size, 3)
    local real_pose_set = torch.Tensor(size, 3)

    for i=1,size do
      -- Get the sample corresponding to current arm pose
      local real_im, pose_real = self:get_one_sample_real()
      local real_im_t, pose_real_t = self:get_one_sample_real_labelled()
      local sim_im, pose_sim = self:get_one_sample()

      -- Generate low dimensional features
      local low_dim_features_real = self:outputLowDimTensor(nil, pose_real)
      local low_dim_features_real_t = self:outputLowDimTensor(nil, pose_real_t)
      local low_dim_features_sim = self:outputLowDimTensor(nil, pose_sim)
      s_sim_set[i] = sim_im
      s_real_set[i] = real_im
      s_real_set_t[i] = real_im_t
      sim_pose_set[i] = low_dim_features_sim
      real_pose_set[i] = low_dim_features_real
      real_pose_set_t[i] = low_dim_features_real_t
    end

    return s_sim_set, s_real_set_t, s_real_set, sim_pose_set, real_pose_set_t, real_pose_set
end


-- Get a batch with a certain size from a dataset
function dctrl:get_testset(size)
    -- TODO: make the initialization of s_set and l_set autonomous
    local s_set = torch.Tensor(size, 3, unpack(self.image_res))
    local l_set = torch.Tensor(size, 3)
    local sim_size = size

    for i=1,size do
      -- Get the sample corresponding to current arm pose
      local image_, label

      image_, label = self:get_one_test_sample()

      -- Generate low dimensional features
      local low_dim_features = self:outputLowDimTensor(nil, label)

      s_set[i] = image_
      l_set[i] = low_dim_features
    end

    return s_set, l_set
end


-- Get one sample from a dataset
function dctrl:get_one_sample_extra()
    -- Randomly select a sample from the dataset
    local index = 1 + math.floor(self.extra_dataset_sample_amount * torch.rand(1)[1])
    if index > self.extra_dataset_sample_amount then
      index = self.extra_dataset_sample_amount
    end

    local image_ = self.extra_dataset_image[index]:clone()
    local label = table.copy(self.extra_dataset_label[index])

    -- local label = self.dataset.label[index]:clone()
    -- local end_effector = {self.dataset.end_effector[index][1], self.dataset.end_effector[index][2]}

    -- Add some noisy factors
    if self.add_noise then
      if torch.uniform() < self.noisy_prob then
        image_ = self:varying_white_balance(image_)
      end
      -- Brightness
      if torch.uniform() < self.noisy_prob then
        --Change brightness by changing the v in hsv sapce
        local im_hsv = image.rgb2hsv(image_)
        local im_v = im_hsv:select(1, 3)
        im_v:mul(1+1.6*(torch.uniform()-0.5))
        image_ = image.hsv2rgb(im_hsv)
        image_ = torch.clamp(image_,0,1)
      end
      -- -- Rotation
      -- if torch.rand(1)[1] < self.noisy_prob then
      --     image_ = image.rotate(image_, 0.2*(torch.rand(1)[1]-0.5))
      --     noise_added = true
      -- end
      -- -- Offset
      -- if torch.rand(1)[1] < self.noisy_prob then
      --     image_ = image.translate(image_, 10*(torch.rand(1)[1]-0.5), 10*(torch.rand(1)[1]-0.5))
      --     noise_added = true
      -- end
      -- -- White noise
      -- if torch.rand(1)[1] < self.noisy_prob then
      --     image_ = self:addRandomNoises(image_, 0.01)
      --     noise_added = true
      -- end
    end

    image_ = self:image_normalization(image_)
    if self.upper_plane then
      -- print("Centre: ", torch.Tensor(label))
      label[3] = label[3] + self.target_h
      -- print("Upper Plane: ", torch.Tensor(label))
    end

    return image_, label
end

-- Get one sample from a dataset
function dctrl:get_one_sample_real_labelled()
    -- Randomly select a sample from the dataset
    local index = 1 + math.floor(self.extra_dataset_sample_amount * torch.rand(1)[1])
    if index > self.extra_dataset_sample_amount then
      index = self.extra_dataset_sample_amount
    end

    local image_ = self.extra_dataset_image[index]:clone()
    local label = table.copy(self.extra_dataset_label[index])

    -- local label = self.dataset.label[index]:clone()
    -- local end_effector = {self.dataset.end_effector[index][1], self.dataset.end_effector[index][2]}

    -- Add some noisy factors
    if self.add_noise then
      if torch.uniform() < self.noisy_prob then
        image_ = self:varying_white_balance(image_)
      end
      -- Brightness
      if torch.uniform() < self.noisy_prob then
        --Change brightness by changing the v in hsv sapce
        local im_hsv = image.rgb2hsv(image_)
        local im_v = im_hsv:select(1, 3)
        -- im_v:mul(1+1.6*(torch.uniform()-0.5))
        im_v:mul(1+0.8*(torch.uniform()-0.5))
        image_ = image.hsv2rgb(im_hsv)
        image_ = torch.clamp(image_,0,1)
      end

      -- -- Rotation
      -- if torch.rand(1)[1] < self.noisy_prob then
      --     image_ = image.rotate(image_, 0.2*(torch.rand(1)[1]-0.5))
      --     noise_added = true
      -- end
      -- -- Offset
      -- if torch.rand(1)[1] < self.noisy_prob then
      --     image_ = image.translate(image_, 10*(torch.rand(1)[1]-0.5), 10*(torch.rand(1)[1]-0.5))
      --     noise_added = true
      -- end
      -- -- White noise
      -- if torch.rand(1)[1] < self.noisy_prob then
      --     image_ = self:addRandomNoises(image_, 0.01)
      --     noise_added = true
      -- end
    end

    image_ = self:image_normalization(image_)
    if self.upper_plane then
      -- print("Centre: ", torch.Tensor(label))
      label[3] = label[3] + self.target_h
      -- print("Upper Plane: ", torch.Tensor(label))
    end

    return image_, label
end

-- Get one sample from a dataset
function dctrl:get_one_sample_real()
    -- Randomly select a sample from the dataset
    local index = 1 + math.floor(self.real_dataset_sample_amount * torch.rand(1)[1])
    if index > self.real_dataset_sample_amount then
      index = self.real_dataset_sample_amount
    end

    local image_ = self.real_dataset_image[index]:clone()
    local label = table.copy(self.real_dataset_label[index])

    -- local label = self.dataset.label[index]:clone()
    -- local end_effector = {self.dataset.end_effector[index][1], self.dataset.end_effector[index][2]}

    -- Add some noisy factors
    if self.add_noise then
      if torch.uniform() < self.noisy_prob then
        image_ = self:varying_white_balance(image_)
      end
      -- Brightness
      if torch.uniform() < self.noisy_prob then
        --Change brightness by changing the v in hsv sapce
        local im_hsv = image.rgb2hsv(image_)
        local im_v = im_hsv:select(1, 3)
        im_v:mul(1+0.8*(torch.uniform()-0.5))
        image_ = image.hsv2rgb(im_hsv)
        image_ = torch.clamp(image_,0,1)
      end
      -- -- Rotation
      -- if torch.rand(1)[1] < self.noisy_prob then
      --     image_ = image.rotate(image_, 0.2*(torch.rand(1)[1]-0.5))
      --     noise_added = true
      -- end
      -- -- Offset
      -- if torch.rand(1)[1] < self.noisy_prob then
      --     image_ = image.translate(image_, 10*(torch.rand(1)[1]-0.5), 10*(torch.rand(1)[1]-0.5))
      --     noise_added = true
      -- end
      -- -- White noise
      -- if torch.rand(1)[1] < self.noisy_prob then
      --     image_ = self:addRandomNoises(image_, 0.01)
      --     noise_added = true
      -- end
    end

    image_ = self:image_normalization(image_)
    if self.upper_plane then
      -- print("Centre: ", torch.Tensor(label))
      label[3] = label[3] + self.target_h
      -- print("Upper Plane: ", torch.Tensor(label))
    end

    return image_, label
end


-- Get a batch with a certain size from a dataset
function dctrl:get_batch_extra(size)
    -- TODO: make the initialization of s_set and l_set autonomous
    local s_set = torch.Tensor(size, 3, unpack(self.image_res))
    local l_set = torch.Tensor(size, 3)

    local sim_size = size
    if self.real_dataset then
        -- sim_size = math.floor(0.5 * size)
        -- sim_size = math.floor(0.15 * size)
        sim_size = 0
    end

    for i=1,size do
      -- Get the sample corresponding to current arm pose
      local image_, label
      if i > sim_size then
        image_, label = self:get_one_sample_real()
      else
        image_, label = self:get_one_sample_extra()
      end

      -- Generate low dimensional features
      local low_dim_features = self:outputLowDimTensor(nil, label)

      s_set[i] = image_
      l_set[i] = low_dim_features
    end

    return s_set, l_set
end

-- Get one sample from a dataset
function dctrl:get_one_sample_e2e()
    -- Randomly select a sample from the dataset
    local index = 1 + math.floor(self.e2e_dataset_sample_amount * torch.rand(1)[1])
    if index > self.e2e_dataset_sample_amount then
      index = self.e2e_dataset_sample_amount
    end

    local image_ = self.e2e_dataset_image[index]:clone()
    local position = table.copy(self.e2e_dataset_label[index])
    local low_dim_s = self.e2e_dataset_s[index]:clone()
    local vel = table.copy(self.e2e_dataset_vel[index])
    -- local arm_pose = table.copy(self.dataset_arm_pose[index])

    -- local label = self.dataset.label[index]:clone()
    -- local end_effector = {self.dataset.end_effector[index][1], self.dataset.end_effector[index][2]}

    -- Add some noisy factors
    if self.add_noise then
      if torch.uniform() < self.noisy_prob then
        image_ = self:varying_white_balance(image_)
      end
      -- Brightness
      if torch.uniform() < self.noisy_prob then
        --Change brightness by changing the v in hsv sapce
        local im_hsv = image.rgb2hsv(image_)
        local im_v = im_hsv:select(1, 3)
        im_v:mul(1+1.6*(torch.uniform()-0.5))
        image_ = image.hsv2rgb(im_hsv)
        image_ = torch.clamp(image_,0,1)
      end
      -- -- Rotation
      -- if torch.rand(1)[1] < self.noisy_prob then
      --     image_ = image.rotate(image_, 0.2*(torch.rand(1)[1]-0.5))
      --     noise_added = true
      -- end
      -- -- Offset
      -- if torch.rand(1)[1] < self.noisy_prob then
      --     image_ = image.translate(image_, 10*(torch.rand(1)[1]-0.5), 10*(torch.rand(1)[1]-0.5))
      --     noise_added = true
      -- end
      -- -- White noise
      -- if torch.rand(1)[1] < self.noisy_prob then
      --     image_ = self:addRandomNoises(image_, 0.01)
      --     noise_added = true
      -- end
    end

    image_ = self:image_normalization(image_)
    -- if self.upper_plane then
    --   -- print("Centre: ", torch.Tensor(label))
    --   position[3] = position[3] + self.target_h
    --   -- print("Upper Plane: ", torch.Tensor(label))
    -- end

    return image_, position, low_dim_s, vel
end

-- Get one sample from a dataset
function dctrl:get_one_sample_e2e_real()
    -- Randomly select a sample from the dataset
    local index = 1 + math.floor(self.e2e_real_dataset_sample_amount * torch.rand(1)[1])
    if index > self.e2e_real_dataset_sample_amount then
      index = self.e2e_real_dataset_sample_amount
    end

    local image_ = self.e2e_real_dataset_image[index]:clone()
    local position = table.copy(self.e2e_real_dataset_label[index])
    local low_dim_s = self.e2e_real_dataset_s[index]:clone()
    local vel = table.copy(self.e2e_real_dataset_vel[index])

    -- local label = self.dataset.label[index]:clone()
    -- local end_effector = {self.dataset.end_effector[index][1], self.dataset.end_effector[index][2]}

    -- Add some noisy factors
    if self.add_noise then
      if torch.uniform() < self.noisy_prob then
        image_ = self:varying_white_balance(image_)
      end
      -- Brightness
      if torch.uniform() < self.noisy_prob then
        --Change brightness by changing the v in hsv sapce
        local im_hsv = image.rgb2hsv(image_)
        local im_v = im_hsv:select(1, 3)
        im_v:mul(1+0.8*(torch.uniform()-0.5))
        image_ = image.hsv2rgb(im_hsv)
        image_ = torch.clamp(image_,0,1)
      end
      -- -- Rotation
      -- if torch.rand(1)[1] < self.noisy_prob then
      --     image_ = image.rotate(image_, 0.2*(torch.rand(1)[1]-0.5))
      --     noise_added = true
      -- end
      -- -- Offset
      -- if torch.rand(1)[1] < self.noisy_prob then
      --     image_ = image.translate(image_, 10*(torch.rand(1)[1]-0.5), 10*(torch.rand(1)[1]-0.5))
      --     noise_added = true
      -- end
      -- -- White noise
      -- if torch.rand(1)[1] < self.noisy_prob then
      --     image_ = self:addRandomNoises(image_, 0.01)
      --     noise_added = true
      -- end
    end

    image_ = self:image_normalization(image_)
    if self.upper_plane then
      -- print("Centre: ", torch.Tensor(label))
      position[3] = position[3] + self.target_h
      -- print("Upper Plane: ", torch.Tensor(label))
    end

    return image_, position, low_dim_s, vel
end

-- TODO: develop a function to get batches for end-to-end fine-tuning
function dctrl:get_batch_e2e(size)
  local s_set = torch.Tensor(size, 3, unpack(self.image_res))
  local l_set = torch.Tensor(size, 3)
  local s_q_set = torch.Tensor(size, 7)
  local v_set = torch.Tensor(size, 7)

  local sim_size = size
  -- if self.e2e_real_dataset and not self.weighted_losses then
  if self.e2e_real_dataset then
      sim_size = math.floor(0.5 * size)
      -- sim_size = math.floor(0.125 * size)
      -- sim_size = 0
  end

  for i=1,size do
    -- Get the sample corresponding to current arm pose
    local image_, position, low_dim_s, vel
    if i > sim_size then
      image_, position, low_dim_s, vel = self:get_one_sample_e2e_real()
    else
      image_, position, low_dim_s, vel = self:get_one_sample_e2e()
    end
    -- Generate low dimensional features
    local low_dim_features = self:outputLowDimTensor(nil, position)
    s_set[i] = image_
    l_set[i] = low_dim_features
    s_q_set[i] = low_dim_s:narrow(1,1,7)
    v_set[i] = torch.Tensor(vel)
  end

  return {s_q_set, s_set}, v_set, l_set
end

-- Get one sample from a dataset
function dctrl:get_one_sample_ctrl(arm_pose)
    -- Randomly select a sample from the dataset
    local index = 1 + math.floor(self.dataset_sample_amount * torch.rand(1)[1])
    if index > self.dataset_sample_amount then
      index = self.dataset_sample_amount
    end

    local s_ = self.dataset_s[index]:clone()
    local label = torch.Tensor(self.dataset_label[index])

    -- local label = self.dataset.label[index]:clone()
    -- local end_effector = {self.dataset.end_effector[index][1], self.dataset.end_effector[index][2]}

    return s_, label
end

-- Get a batch with a certain size from a dataset
function dctrl:get_batch_ctrl(size)
    -- TODO: make the initialization of s_set and l_set autonomous
    local s_set = torch.Tensor(size, 10)
    local l_set = torch.Tensor(size, 7)

    for i=1,size do
      -- Get the sample corresponding to current arm pose
      local s_, label = self:get_one_sample_ctrl()
      -- Generate low dimensional features
      -- local low_dim_features = self:outputLowDimTensor(nil, label)

      s_set[i] = s_
      l_set[i] = label
    end

    return s_set, l_set
end

function dctrl:setCameraPose(random)
  local camera_pose = table.copy(self.right_arm_init_pose)
  if random then
    for i=1,self.n_joint do
      camera_pose[i] = camera_pose[i] + (torch.uniform() - 0.5) * self.camera_pose_range
    end
  end

  self:setArmTargetPose('right',camera_pose)
end

function dctrl:randTargetOrientation()
  local orientation_z = torch.uniform() * 2 * math.pi
  return {0.0,0.0,orientation_z}
end

function dctrl:randTablePose(random)
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

function dctrl:randClutters()
  -- Randomly set the tabletop clutters

  -- number of clutter objects
  local objects_display = {}
  local clutters_h = {}
  local objects_remove = {}
  for i=1,self.n_clutters do
    if torch.uniform() < 0.5 then
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

function dctrl:removeObjects(clutters)
  local n = #clutters
  for i=1,n do
    self.vrep:setObjectPosition(clutters[i], self.table, {0.0, 0.0, 0.0})
  end
end

function dctrl:randObjects(clutters, clutters_h)
  local n = #clutters
  for i=1,n do
    -- self.vrep:scaleObjectSize(clutters[i], {0.1*torch.uniform(),0.1*torch.uniform()})
    self.vrep:setObjectPosition(clutters[i], self.table_top, {(torch.uniform()-0.35)*0.9, (torch.uniform()-0.5)*0.6, clutters_h[i]})
    self.vrep:setObjectOrientation(clutters[i], self.table_top, {0.0,0.0,torch.uniform() * 2 * math.pi})
  end
  self.vrep:setObjectOrientation(self.clutters_handles[5], -1, {torch.uniform()*2*math.pi,0.5*math.pi,0.0})
  -- self.vrep:setObjectOrientation(self.clutters_handles[5], -1, {torch.uniform()*2*math.pi,0.0,0.0})
end

function dctrl:randTargetSize()

end

-- Generate a dataset for perception
function dctrl:generate_perception_dataset(display)
  -- Initialize dataset parameters

  -- local self_collision_allowance = {0.88, 0.88, 0.96}
  -- local pos_min = {0.1, -0.2625, 0.0325}
  -- local pos_max = {0.7, 0.2625, 0.1825}
  local pos_min = table.copy(self.target_pose_min)
  local pos_max = table.copy(self.target_pose_max)
  local average_accuracy = 0.01
  -- local average_accuracy = 0.2

  print("Generating samples ...... ")

  local target = {0.1, -0.2625, 0.0325}

  local num_x = math.floor((pos_max[1] - pos_min[1])/average_accuracy + 0.5)
  local num_y = math.floor((pos_max[2] - pos_min[2])/average_accuracy + 0.5)
  local num_z = math.floor((pos_max[3] - pos_min[3])/average_accuracy + 0.5)
  print("num_x: ", num_x)
  print("num_y: ", num_y)
  print("num_y: ", num_z)
  local sample_index = 0
  local h_start = 1
  for k=0,num_z do
    if k==1 then
      h_start = sample_index + 1
      print("h start: ", h_start)
    end
    local image_ = {} -- images 400x400
    local image400 = {}
    local label = {} -- labels object pose
    sample_index = 0
    for i=1,num_x do
      for j=1,num_y do
        target[1] = pos_min[1] + (i-1 + torch.rand(1)[1]) * average_accuracy
        if target[1] > pos_max[1] then
          target[1] = pos_max[1]
        end
        target[2] = pos_min[2] + (j-1 + torch.rand(1)[1]) * average_accuracy
        if target[2] > pos_max[2] then
          target[2] = pos_max[2]
        end
        target[3] = pos_min[3] + (k-1 + torch.rand(1)[1]) * average_accuracy
        if target[3] > pos_max[3] then
          target[3] = pos_max[3]
        end
        self:setCameraPose(true)
        self:randTablePose(true)
        self:randClutters()
        -- TODO: Randomize shape colors including background, table, objects and even robot body


        local table_position = self.vrep:getObjectPosition(self.table_top, self.robot_base)
        -- print("Table Position: ", torch.Tensor(table_position))
        if k==0 or target[3] < table_position[3] then
          target[3] = table_position[3] + 0.0325
        end
        local pose = self:initArmPose()
        local target_orien = self:randTargetOrientation()
        self.vrep:setObjectOrientation(self.cuboid, self.robot_base, target_orien)
        self.vrep:setObjectPosition(self.cuboid, self.robot_base, target)
        local desired_pose = self.vrep:getDesiredConfig(0.2)
        if desired_pose~=nil then
          if #desired_pose~=0 then
            -- print("*******",i)
            pose = self:generateRandPose(pose, desired_pose)
          end
        end

        local position = self.vrep:getObjectPosition(self.cuboid, self.robot_base)
        -- local position_e = self.vrep:getObjectPosition(self.cuboid, self.robot_base)
        -- local position_t2e = self.vrep:getObjectPosition(self.cuboid, self.robot_base)


        if self.image_format == 'RGB' then
          im = self.vrep:getRGBImage(self.right_camera)
        else
          im = self.vrep:getGreyImage(self.right_camera)
        end
        -- TODO develop codes to crop and resize the image
        local cropped_image = image.crop(im,120,20,420,320)
        local cropped_image400 = image.crop(im,0,0,400,400)
        -- print("Image size: ", cropped_image:size())

        -- Add into the dataset
        sample_index = sample_index + 1
        image_[sample_index] = cropped_image:float():div(255)
        image400[sample_index] = cropped_image400:float():div(255)
        -- print(image_[sample_index]:max())
        label[sample_index] = position

        -- Visualize images and output results
        -- if display then
        win_input1 = image.display{image=image_[sample_index], win=win_input1}
        win_input2 = image.display{image=image400[sample_index], win=win_input2}
        -- end
        print("=====================================")
        print("Sample Index: ", sample_index)
        print("Label: ", torch.Tensor(label[sample_index]))

        if sample_index%50 == 0 then
            print("Samples Collected: ", sample_index)
        end
      end
    end


    -- Get the dimensionality of each element in the dataset
    local image_dim = image_[1]:size()
    local label_dim = #label[1]

    -- Save the dataset to a t7 file
    local filename_ = "/run/user/1000/gvfs/smb-share:server=hpc-fs.qut.edu.au,share=n9314181/table_top_blue_box_dataset" .. average_accuracy .. "_" .. k
    torch.save(filename_ .. ".t7", {image = image_,
                            image400 = image400,
                            label = label,
                            image_dim = image_dim,
                            label_dim = label_dim,
                            sample_amount = sample_index,
                            average_accuracy = average_accuracy,
                            num_x = num_x,
                            num_y = num_y,
                            num_z = num_z,
                            h_start = h_start})
    print('Image dataset generated:', filename_ .. '.t7')
    print("Dataset Property: \n")
    print("image_dim: \n", image_dim)
    print("label_dim: \n", label_dim)
    print("Samples Collected: ", sample_index)

    io.flush()
    collectgarbage()
  end

  print("num_x: ", num_x)
  print("num_y: ", num_y)
  print("num_z: ", num_z)
  print("accuracy: ", average_accuracy)

end

-- Generate a dataset for perception
function dctrl:generate_perception_dataset_dr(display)
  -- Initialize dataset parameters

  -- local self_collision_allowance = {0.88, 0.88, 0.96}
  -- local pos_min = {0.1, -0.2625, 0.0325}
  -- local pos_max = {0.7, 0.2625, 0.1825}
  local pos_min = table.copy(self.target_pose_min)
  local pos_max = table.copy(self.target_pose_max)
  local average_accuracy = 0.01
  -- local average_accuracy = 0.2

  print("Generating samples ...... ")

  local target = {0.1, -0.2625, 0.0325}

  local num_x = math.floor((pos_max[1] - pos_min[1])/average_accuracy + 0.5)
  local num_y = math.floor((pos_max[2] - pos_min[2])/average_accuracy + 0.5)
  -- local num_z = math.floor((pos_max[3] - pos_min[3])/average_accuracy + 0.5)
  local num_z = 4
  print("num_x: ", num_x)
  print("num_y: ", num_y)
  print("num_z: ", num_z)
  local sample_index = 0
  for k=0,num_z do
    local image_ = {} -- images 400x400
    local image400 = {}
    local label = {} -- labels object pose
    sample_index = 0
    for i=1,num_x do
      for j=1,num_y do
        target[1] = pos_min[1] + (i-1 + torch.rand(1)[1]) * average_accuracy
        if target[1] > pos_max[1] then
          target[1] = pos_max[1]
        end
        target[2] = pos_min[2] + (j-1 + torch.rand(1)[1]) * average_accuracy
        if target[2] > pos_max[2] then
          target[2] = pos_max[2]
        end
        -- target[3] = pos_min[3] + (k-1 + torch.rand(1)[1]) * average_accuracy
        -- if target[3] > pos_max[3] then
        --   target[3] = pos_max[3]
        -- end

        -- Randomize camera pose and FoV
        self:setCameraPose(true)
        self.vrep:randFoV()

        -- Randomize table pose
        -- self:randTablePose(true)
        local table_position = self.vrep:getObjectPosition(self.table_top, self.robot_base)
        -- print("Table Position: ", torch.Tensor(table_position))
        -- if k==0 or target[3] < table_position[3] then
        target[3] = table_position[3] + 0.0325
        -- end
        local pose = self:initArmPose()

        -- Randomize target orientation
        local target_orien = self:randTargetOrientation()
        self.vrep:setObjectOrientation(self.cuboid, self.robot_base, target_orien)
        self.vrep:setObjectPosition(self.cuboid, self.robot_base, target)

        -- Randomize scene colors
        self.vrep:randSceneColor()

        -- Randomize clutter colors and poses
        self.vrep:randClutter()

        -- Randomize left arm pose
        local desired_pose = self.vrep:getDesiredConfig(0.2)
        if desired_pose~=nil then
          if #desired_pose~=0 then
            -- print("*******",i)
            pose = self:generateRandPose(pose, desired_pose)
          end
        end

        local position = self.vrep:getObjectPosition(self.cuboid, self.robot_base)
        -- local position_e = self.vrep:getObjectPosition(self.cuboid, self.robot_base)
        -- local position_t2e = self.vrep:getObjectPosition(self.cuboid, self.robot_base)


        if self.image_format == 'RGB' then
          im = self.vrep:getRGBImage(self.right_camera)
        else
          im = self.vrep:getGreyImage(self.right_camera)
        end
        -- TODO develop codes to crop and resize the image
        local cropped_image = image.crop(im,120,20,420,320)
        local cropped_image400 = image.crop(im,0,0,400,400)
        -- print("Image size: ", cropped_image:size())

        -- Add into the dataset
        sample_index = sample_index + 1
        image_[sample_index] = cropped_image:float():div(255)
        image400[sample_index] = cropped_image400:float():div(255)
        -- print(image_[sample_index]:max())
        label[sample_index] = position

        -- Visualize images and output results
        -- if display then
        win_input1 = image.display{image=image_[sample_index], win=win_input1}
        win_input2 = image.display{image=image400[sample_index], win=win_input2}
        -- end
        print("=====================================")
        print("Sample Index: ", sample_index)
        print("Label: ", torch.Tensor(label[sample_index]))

        if sample_index%50 == 0 then
            print("Samples Collected: ", sample_index)
        end
      end
    end


    -- Get the dimensionality of each element in the dataset
    local image_dim = image_[1]:size()
    local label_dim = #label[1]

    -- Save the dataset to a t7 file
    local filename_ = "/run/user/1000/gvfs/smb-share:server=hpc-fs.qut.edu.au,share=n9314181/table_top_blue_box_dataset_dr_no_poseR" .. average_accuracy .. "_" .. k
    torch.save(filename_ .. ".t7", {image = image_,
                            image400 = image400,
                            label = label,
                            image_dim = image_dim,
                            label_dim = label_dim,
                            sample_amount = sample_index,
                            average_accuracy = average_accuracy,
                            num_x = num_x,
                            num_y = num_y,
                            num_z = num_z})
    print('Image dataset generated:', filename_ .. '.t7')
    print("Dataset Property: \n")
    print("image_dim: \n", image_dim)
    print("label_dim: \n", label_dim)
    print("Samples Collected: ", sample_index)

    io.flush()
    collectgarbage()
  end

  print("num_x: ", num_x)
  print("num_y: ", num_y)
  print("num_z: ", num_z)
  print("accuracy: ", average_accuracy)

end


-- Generate a dataset for perception
function dctrl:generate_control_dataset(display)

  -- local self_collision_allowance = {0.88, 0.88, 0.96}
  -- local pos_min = {0.1, -0.2625, 0.025}
  -- local pos_max = {0.7, 0.2625, 0.025}
  local pos_min = table.copy(self.target_pose_min)
  local pos_max = table.copy(self.target_pose_max)
  local average_accuracy = 0.01
  -- local average_accuracy = 0.2
  -- local rand_scalar = 2 * average_accuracy

  self:setCameraPose(false)

  print("Generating samples ...... ")

  local target = {0.1, -0.2625, 0.025}

  local num_x = math.floor((pos_max[1] - pos_min[1])/average_accuracy + 0.5)
  local num_y = math.floor((pos_max[2] - pos_min[2])/average_accuracy + 0.5)
  local num_z = math.floor((pos_max[3] - pos_min[3])/average_accuracy + 0.5)
  print("num_x: ", num_x)
  print("num_y: ", num_y)
  print("num_z: ", num_z)
  for k=0,num_z do
    local label_ = {}
    local target_pose_ = {}
    local desired_pose_ = {}
    local initial_arm_pose_ = {}
    local current_arm_pose_ = {}
    local vel_cmd_ = {}
    local low_dim_s_ = {}
    local label_v_ = {}

    local low_dim_s_e = {} -- the low dim state for the ultimate 3DoF end-effector position
    local target_pose_e = {}
    local no_config_target = {}
    local no_config_low_dim_s = {}
    local no_config_vel = {}

    local sample_index = 0
    local no_config_index = 0
    for i=1,num_x do
      for j=1,num_y do
        target[1] = pos_min[1] + (i-1 + torch.rand(1)[1]) * average_accuracy
        if target[1] > pos_max[1] then
          target[1] = pos_max[1]
        end
        target[2] = pos_min[2] + (j-1 + torch.rand(1)[1]) * average_accuracy
        if target[2] > pos_max[2] then
          target[2] = pos_max[2]
        end
        target[3] = pos_min[3] + (k-1 + torch.rand(1)[1]) * average_accuracy
        if target[3] > pos_max[3] then
          target[3] = pos_max[3]
        end

        self:randTablePose(true)
        local table_position = self.vrep:getObjectPosition(self.table_top, self.robot_base)
        -- print("Table Position: ", torch.Tensor(table_position))
        if k==0 or target[3] < table_position[3] then
          target[3] = table_position[3] + 0.0325
        end
        local init_pose = self:initArmPose()
        local target_orien = self:randTargetOrientation()
        self.vrep:setObjectOrientation(self.cuboid, self.robot_base, target_orien)
        self.vrep:setObjectPosition(self.cuboid, self.robot_base, target)

        local desired_pose = self.vrep:getDesiredConfig(0.2)
        if desired_pose == nil then
          local no_target = self.vrep:getObjectPosition(self.cuboid, self.robot_base)
          no_target[3] = no_target[3] + self.target_h
          no_config_index = no_config_index + 1
          no_config_target[no_config_index] = table.copy(no_target)
          no_config_low_dim_s[no_config_index] = self:outputLowDimTensor(init_pose, no_target)
          no_config_vel[no_config_index] = {0.0,0.0,0.0,0.0,0.0,0.0,0.0}
          -- print("no_config_target: ", torch.Tensor(no_config_target[no_config_index]))
          -- print("no_config_target low s: ", torch.Tensor(no_config_low_dim_s[no_config_index]))

        elseif #desired_pose == 0 then
          local no_target = self.vrep:getObjectPosition(self.cuboid, self.robot_base)
          no_target[3] = no_target[3] + self.target_h
          no_config_index = no_config_index + 1
          no_config_target[no_config_index] = table.copy(no_target)
          no_config_low_dim_s[no_config_index] = self:outputLowDimTensor(init_pose, no_target)
          no_config_vel[no_config_index] = {0.0,0.0,0.0,0.0,0.0,0.0,0.0}
          -- print("no_config_target: ", torch.Tensor(no_config_target[no_config_index]))
          -- print("no_config_target low s: ", torch.Tensor(no_config_low_dim_s[no_config_index]))

        else
          -- print("*******",i)
          local pose = self:generateRandPose(init_pose, desired_pose)
          local collision, reached = self:detectCollisions()
          while collision and not reached do
            print("Initial collision happened, filter out !!!!")
            pose = self:generateRandPose(init_pose, desired_pose)
            collision, reached = self:detectCollisions()
          end

          local position = self.vrep:getObjectPosition(self.cuboid, self.robot_base)
          position[3] = position[3] + self.target_h
          pose = self:getArmPose('left')
          -- local position_t2e = self.vrep:getObjectPosition(self.cuboid, self.robot_base)

          -- Add into the dataset
          sample_index = sample_index + 1
          -- image_[sample_index] = cropped_image:float():div(255)
          -- print(image_[sample_index]:max())
          label_[sample_index] = table.copy(position)
          desired_pose_[sample_index] = table.copy(desired_pose)
          target_pose_[sample_index] = table.copy(position)
          initial_arm_pose_[sample_index] = table.copy(pose)

          -- Visualize images and output results
          -- if display then
          -- win_input1 = image.display{image=image_[sample_index], win=win_input1}
            -- win_input2 = image.display{image=cropped_image, win=win_input2}
          -- end

          local termination = false
          local completion = false
          -- local screen, reward, terminal, completion, low_dim_features, configuration = self:picking_sim(1, true, false)
          local current_arm_pose_temp = {}
          local vel_cmd_temp = {}
          local low_dim_s_temp = {}
          local image_v_temp = {}
          local label_v_temp = {}
          local v_sample_index = 0
          self.vrep:enableJointVelocityCtrl()
          local step_cost = 0
          while not termination do
            step_cost = step_cost + 1
            if step_cost > self.max_step then
              break
            end
            local vel = {}
            for i=1,self.n_joint do
              vel[i] = desired_pose[i] - self.left_joint_position[i]
              if vel[i] > 1.0 then
                vel[i] = 1.0
              elseif vel[i] < -1.0 then
                vel[i] = -1.0
              end
            end
            local vel_sum = torch.Tensor(vel):abs():sum()
            -- print("vel sum: ", vel_sum)
            -- if vel_sum < 0.01 then
            if vel_sum < 0.01 then
              vel = {0.0,0.0,0.0,0.0,0.0,0.0,0.0}
              termination = true
              completion = true
            end

            local low_dim_features = self:outputLowDimTensor(self.left_joint_position, position)
            v_sample_index = v_sample_index + 1
            current_arm_pose_temp[v_sample_index] = table.copy(self.left_joint_position)
            vel_cmd_temp[v_sample_index] = table.copy(vel)
            low_dim_s_temp[v_sample_index] = low_dim_features
            -- image_v_temp[v_sample_index] = im
            label_v_temp[v_sample_index] = table.copy(vel)
            -- print("pose: ", torch.Tensor(current_arm_pose_temp[v_sample_index]))
            -- print("low s: ", low_dim_s_temp[v_sample_index])
            -- print("vel cmd: ", torch.Tensor(label_v_temp[v_sample_index]))
            -- win_input1 = image.display{image=image_v_temp[v_sample_index], win=win_input1}

            self:ctrlSpeedContinuous(vel)
            local collision, reached = self:getMeasurements()
            -- print("collision: ", collision)
            -- print("reached: ", reached)
            if collision then
              termination = true
              completion = false
              print("====== Hit obstacles ======")
              print("k: ", k)
              print("Sample: ", sample_index)

            end

            -- print("termination: ", termination)
            -- print("completion: ", completion)
            -- print("low dim features: ", low_dim_features)
            -- print("Step cost: ", step - ttttt)
            -- ttttt = step

          end
          local position_e = self.vrep:getObjectPosition(self.left_gripper, self.robot_base)
          target_pose_e[sample_index] = table.copy(position_e)
          self.vrep:enableJointPositionCtrl()

          if completion then
            -- print("Final reward: ", reward)
            local n_v_sample = #current_arm_pose_
            for i=1,v_sample_index do
              current_arm_pose_[n_v_sample+i] = current_arm_pose_temp[i]
              vel_cmd_[n_v_sample+i] = vel_cmd_temp[i]
              low_dim_s_[n_v_sample+i] = low_dim_s_temp[i]
              -- image_v_[n_v_sample+i] = image_v_temp[i]
              label_v_[n_v_sample+i] = label_v_temp[i]
              low_dim_s_e[n_v_sample+i] = self:outputLowDimTensor(current_arm_pose_temp[i], position_e)
              -- print("vel low dim s: ", low_dim_s_[n_v_sample+i])
              -- print("vel low dim s e: ", low_dim_s_e[n_v_sample+i])
              -- print("vel: ", torch.Tensor(label_v_[n_v_sample+i]))

            end
            -- print("pose: ", torch.Tensor(current_arm_pose_[n_v_sample+v_sample_index]))
            -- print("low s: ", low_dim_s_[n_v_sample+v_sample_index])
            -- print("vel cmd: ", torch.Tensor(label_v_[n_v_sample+v_sample_index]))
          end


          -- Visualize images and output results
          -- if display then
          -- win_input1 = image.display{image=image_[sample_index], win=win_input1}
            -- win_input2 = image.display{image=cropped_image, win=win_input2}
          -- end
          print("=====================================")
          print("Case Index: ", sample_index)
          print("Sample Index: ", #current_arm_pose_)
          -- print("Label V: ", torch.Tensor(label_v_[#label_v_]))

          if sample_index%50 == 0 then
              print("Cases Collected: ", sample_index)
              print("Sample Collected: ", #low_dim_s_)
              local filename_0 = "/run/user/1000/gvfs/smb-share:server=hpc-fs.qut.edu.au,share=n9314181/case_dataset"
              filename_0 = filename_0 .. average_accuracy .. "_" .. k
              torch.save(filename_0 .. ".t7", {object_pose = target_pose_,
                                      end_effector_pose = target_pose_e,
                                      desired_pose = desired_pose_,
                                      init_pose = initial_arm_pose_,
                                      sample_amount = sample_index,
                                      average_accuracy = average_accuracy,
                                      num_x = num_x,
                                      num_y = num_y,
                                      num_z = num_z})

              filename_0 = "/run/user/1000/gvfs/smb-share:server=hpc-fs.qut.edu.au,share=n9314181/control_dataset"
              filename_0 = filename_0 .. average_accuracy .. "_" .. k
              torch.save(filename_0 .. ".t7", {object_pose = target_pose_,
                                      end_effector_pose = target_pose_e,
                                      desired_pose = desired_pose_,
                                      init_pose = initial_arm_pose_,
                                      arm_pose = current_arm_pose_,
                                      low_dim_s = low_dim_s_,
                                      low_dim_s_e = low_dim_s_e,
                                      label = label_v_,
                                      vel_cmd = vel_cmd_,
                                      sample_amount = #low_dim_s_})

              filename_0 = "/run/user/1000/gvfs/smb-share:server=hpc-fs.qut.edu.au,share=n9314181/no_config_cases"
              filename_0 = filename_0 .. average_accuracy .. "_" .. k
              torch.save(filename_0 .. ".t7", {target_pose = no_config_target,
                                      low_dim_s = no_config_low_dim_s,
                                      label = no_config_vel,
                                      vel_cmd = no_config_vel,
                                      sample_amount = no_config_index})

          end
        end

      end
    end


    -- Get the dimensionality of each element in the dataset
    -- local image_dim = image_[1]:size()
    local label_dim = #label_[1]
    -- Save the dataset to a t7 file
    local filename_ = "/run/user/1000/gvfs/smb-share:server=hpc-fs.qut.edu.au,share=n9314181/case_dataset"
    filename_ = filename_ .. average_accuracy .. "_" .. k
    torch.save(filename_ .. ".t7", {object_pose = target_pose_,
                            end_effector_pose = target_pose_e,
                            desired_pose = desired_pose_,
                            init_pose = initial_arm_pose_,
                            sample_amount = sample_index,
                            average_accuracy = average_accuracy,
                            num_x = num_x,
                            num_y = num_y,
                            num_z = num_z})
    print('Case dataset generated:', filename_ .. '.t7')

    filename_ = "/run/user/1000/gvfs/smb-share:server=hpc-fs.qut.edu.au,share=n9314181/control_dataset"
    filename_ = filename_ .. average_accuracy .. "_" .. k
    torch.save(filename_ .. ".t7", {object_pose = target_pose_,
                            end_effector_pose = target_pose_e,
                            desired_pose = desired_pose_,
                            init_pose = initial_arm_pose_,
                            arm_pose = current_arm_pose_,
                            low_dim_s = low_dim_s_,
                            low_dim_s_e = low_dim_s_e,
                            label = label_v_,
                            vel_cmd = vel_cmd_,
                            sample_amount = #low_dim_s_})
    print('Control dataset generated:', filename_ .. '.t7')

    filename_ = "/run/user/1000/gvfs/smb-share:server=hpc-fs.qut.edu.au,share=n9314181/no_config_cases"
    filename_ = filename_ .. average_accuracy .. "_" .. k
    torch.save(filename_ .. ".t7", {target_pose = no_config_target,
                            low_dim_s = no_config_low_dim_s,
                            label = no_config_vel,
                            vel_cmd = no_config_vel,
                            sample_amount = no_config_index})

    print('No config case dataset generated:', filename_ .. '.t7')
    print("Dataset Property: \n")
    -- print("image_dim: \n", image_dim)
    print("label_dim: \n", label_dim)
    print("case amount: ", sample_index)
    print("vel simple amount: ", #low_dim_s_)
    print("no config case amount: ", no_config_index)
    print("accuracy: ", average_accuracy)
    print("num_x: ", num_x)
    print("num_y: ", num_y)
    print("num_z: ", num_z)
    io.flush()
    collectgarbage()
  end

end


-- Generate a dataset for perception
function dctrl:generate_control_dataset_off_vel(display)

  -- local self_collision_allowance = {0.88, 0.88, 0.96}
  -- local pos_min = {0.1, -0.2625, 0.025}
  -- local pos_max = {0.7, 0.2625, 0.025}
  local pos_min = table.copy(self.target_pose_min)
  local pos_max = table.copy(self.target_pose_max)
  local average_accuracy = 0.03
  -- local average_accuracy = 0.2
  -- local rand_scalar = 2 * average_accuracy

  self:setCameraPose(false)

  print("Generating samples ...... ")

  local target = {0.1, -0.2625, 0.025}

  local num_x = math.floor((pos_max[1] - pos_min[1])/average_accuracy + 0.5)
  local num_y = math.floor((pos_max[2] - pos_min[2])/average_accuracy + 0.5)
  local num_z = math.floor((pos_max[3] - pos_min[3])/average_accuracy + 0.5)
  print("num_x: ", num_x)
  print("num_y: ", num_y)
  print("num_z: ", num_z)
  for k=0,num_z do
    local label_ = {}
    local target_pose_ = {}
    local desired_pose_ = {}
    local initial_arm_pose_ = {}
    local current_arm_pose_ = {}
    local vel_cmd_ = {}
    local low_dim_s_ = {}
    local label_v_ = {}

    local low_dim_s_e = {} -- the low dim state for the ultimate 3DoF end-effector position
    local target_pose_e = {}
    local no_config_target = {}
    local no_config_low_dim_s = {}
    local no_config_vel = {}

    local sample_index = 0
    local no_config_index = 0
    for i=1,num_x do
      for j=1,num_y do
        target[1] = pos_min[1] + (i-1 + torch.rand(1)[1]) * average_accuracy
        if target[1] > pos_max[1] then
          target[1] = pos_max[1]
        end
        target[2] = pos_min[2] + (j-1 + torch.rand(1)[1]) * average_accuracy
        if target[2] > pos_max[2] then
          target[2] = pos_max[2]
        end
        target[3] = pos_min[3] + (k-1 + torch.rand(1)[1]) * average_accuracy
        if target[3] > pos_max[3] then
          target[3] = pos_max[3]
        end

        self:randTablePose(true)
        local table_position = self.vrep:getObjectPosition(self.table_top, self.robot_base)
        -- print("Table Position: ", torch.Tensor(table_position))
        if k==0 or target[3] < table_position[3] then
          target[3] = table_position[3] + 0.0325
        end
        local init_pose = self:initArmPose()
        local target_orien = self:randTargetOrientation()
        self.vrep:setObjectOrientation(self.cuboid, self.robot_base, target_orien)
        self.vrep:setObjectPosition(self.cuboid, self.robot_base, target)

        local desired_pose = self.vrep:getDesiredConfig(0.2)
        if desired_pose == nil then
          local no_target = self.vrep:getObjectPosition(self.cuboid, self.robot_base)
          no_target[3] = no_target[3] + self.target_h
          no_config_index = no_config_index + 1
          no_config_target[no_config_index] = table.copy(no_target)
          no_config_low_dim_s[no_config_index] = self:outputLowDimTensor(init_pose, no_target)
          no_config_vel[no_config_index] = {0.0,0.0,0.0,0.0,0.0,0.0,0.0}
          -- print("no_config_target: ", torch.Tensor(no_config_target[no_config_index]))
          -- print("no_config_target low s: ", torch.Tensor(no_config_low_dim_s[no_config_index]))

        elseif #desired_pose == 0 then
          local no_target = self.vrep:getObjectPosition(self.cuboid, self.robot_base)
          no_target[3] = no_target[3] + self.target_h
          no_config_index = no_config_index + 1
          no_config_target[no_config_index] = table.copy(no_target)
          no_config_low_dim_s[no_config_index] = self:outputLowDimTensor(init_pose, no_target)
          no_config_vel[no_config_index] = {0.0,0.0,0.0,0.0,0.0,0.0,0.0}
          -- print("no_config_target: ", torch.Tensor(no_config_target[no_config_index]))
          -- print("no_config_target low s: ", torch.Tensor(no_config_low_dim_s[no_config_index]))

        else
          -- print("*******",i)
          local pose = self:generateRandPose(init_pose, desired_pose)
          local collision, reached = self:detectCollisions()
          while collision and not reached do
            print("Initial collision happened, filter out !!!!")
            pose = self:generateRandPose(init_pose, desired_pose)
            collision, reached = self:detectCollisions()
          end

          local position = self.vrep:getObjectPosition(self.cuboid, self.robot_base)
          position[3] = position[3] + self.target_h
          pose = self:getArmPose('left')
          -- local position_t2e = self.vrep:getObjectPosition(self.cuboid, self.robot_base)

          -- Add into the dataset
          sample_index = sample_index + 1
          -- image_[sample_index] = cropped_image:float():div(255)
          -- print(image_[sample_index]:max())
          label_[sample_index] = table.copy(position)
          desired_pose_[sample_index] = table.copy(desired_pose)
          target_pose_[sample_index] = table.copy(position)
          initial_arm_pose_[sample_index] = table.copy(pose)

          -- Visualize images and output results
          -- if display then
          -- win_input1 = image.display{image=image_[sample_index], win=win_input1}
            -- win_input2 = image.display{image=cropped_image, win=win_input2}
          -- end

          local current_arm_pose_temp = {}
          local vel_cmd_temp = {}
          local low_dim_s_temp = {}
          local image_v_temp = {}
          local label_v_temp = {}
          local v_sample_index = 0
          local previous_vel = {0.0,0.0,0.0,0.0,0.0,0.0,0.0}
          local previous_pos = table.copy(pose)
          local next_pose = table.copy(pose)
          local termination = false
          local completion = false
          while not termination do
            local vel = {}
            for i=1,self.n_joint do
              vel[i] = desired_pose[i] - previous_pos[i]
              if vel[i] > 1.0 then
                vel[i] = 1.0
              elseif vel[i] < -1.0 then
                vel[i] = -1.0
              end
            end
            local vel_sum = torch.Tensor(vel):abs():sum()
            -- print("vel sum: ", vel_sum)
            -- if vel_sum < 0.01 then
            if vel_sum < 0.01 then
              vel = {0.0,0.0,0.0,0.0,0.0,0.0,0.0}
              termination = true
              -- completion = true
            end

            local low_dim_features = self:outputLowDimTensor(previous_pos, position)
            v_sample_index = v_sample_index + 1
            current_arm_pose_temp[v_sample_index] = table.copy(previous_pos)
            vel_cmd_temp[v_sample_index] = table.copy(vel)
            low_dim_s_temp[v_sample_index] = low_dim_features
            -- image_v_temp[v_sample_index] = im
            label_v_temp[v_sample_index] = table.copy(vel)
            -- print("pose: ", torch.Tensor(current_arm_pose_temp[v_sample_index]))
            -- print("low s: ", low_dim_s_temp[v_sample_index])
            -- print("vel cmd: ", torch.Tensor(label_v_temp[v_sample_index]))
            -- win_input1 = image.display{image=image_v_temp[v_sample_index], win=win_input1}

            -- self:ctrlSpeedContinuous(vel)
            -- local current_vel = self:getArmVelocity('left')
            -- print("Velocit Cmd: ", torch.Tensor(vel))
            -- print("Current Vel: ", torch.Tensor(current_vel))
            for i=1,7 do
              local real_vel = vel[i] * (1 + 0.1 * (torch.uniform() - 0.5))
              next_pose[i] = previous_pos[i] + 0.5 * (real_vel + previous_vel[i]) * self.ctrl_freq_t * 0.001 * (1 + 0.1 * (torch.uniform() - 0.5))
              previous_vel[i] = real_vel
              previous_pos[i] = next_pose[i]
            end
            -- self:setArmTargetPose('left',next_pose)
            -- for i=1,100 do
            --   if torch.Tensor(self:getArmVelocity('left')):abs():sum() < 0.01 then
            --     print(i)
            --     break
            --   elseif i==100 then
            --     print("Reaching failed, filter out!!!")
            --   end
            -- end

          end
          self:setArmTargetPose('left',next_pose)
          for i=1,100 do
            if torch.Tensor(self:getArmVelocity('left')):abs():sum() < 0.001 then
              -- print(i)
              completion = true
              break
            elseif i==100 then
              print("Reaching failed, filter out!!!")
            end
          end
          local collision, reached = self:getMeasurements()
          if collision then
            completion = false
            print("====== Hit obstacles ======")
            print("k: ", k)
            print("Sample: ", sample_index)
          end

          local position_e = self.vrep:getObjectPosition(self.left_gripper, self.robot_base)
          target_pose_e[sample_index] = table.copy(position_e)

          if completion then
            -- print("Final reward: ", reward)
            local n_v_sample = #current_arm_pose_
            for i=1,v_sample_index do
              current_arm_pose_[n_v_sample+i] = current_arm_pose_temp[i]
              vel_cmd_[n_v_sample+i] = vel_cmd_temp[i]
              low_dim_s_[n_v_sample+i] = low_dim_s_temp[i]
              -- image_v_[n_v_sample+i] = image_v_temp[i]
              label_v_[n_v_sample+i] = label_v_temp[i]
              low_dim_s_e[n_v_sample+i] = self:outputLowDimTensor(current_arm_pose_temp[i], position_e)
              -- print("vel low dim s: ", low_dim_s_[n_v_sample+i])
              -- print("vel low dim s e: ", low_dim_s_e[n_v_sample+i])
              -- print("vel: ", torch.Tensor(label_v_[n_v_sample+i]))

            end
            -- print("pose: ", torch.Tensor(current_arm_pose_[n_v_sample+v_sample_index]))
            -- print("low s: ", low_dim_s_[n_v_sample+v_sample_index])
            -- print("vel cmd: ", torch.Tensor(label_v_[n_v_sample+v_sample_index]))
          end


          -- Visualize images and output results
          -- if display then
          -- win_input1 = image.display{image=image_[sample_index], win=win_input1}
            -- win_input2 = image.display{image=cropped_image, win=win_input2}
          -- end
          print("=====================================")
          print("Case Index: ", sample_index)
          print("Sample Index: ", #current_arm_pose_)
          -- print("Label V: ", torch.Tensor(label_v_[#label_v_]))

          if sample_index%50 == 0 then
              print("Cases Collected: ", sample_index)
              print("Sample Collected: ", #low_dim_s_)
              local filename_0 = "/run/user/1000/gvfs/smb-share:server=hpc-fs.qut.edu.au,share=n9314181/case_dataset_off_vel"
              filename_0 = filename_0 .. average_accuracy .. "_" .. k
              torch.save(filename_0 .. ".t7", {object_pose = target_pose_,
                                      end_effector_pose = target_pose_e,
                                      desired_pose = desired_pose_,
                                      init_pose = initial_arm_pose_,
                                      sample_amount = sample_index,
                                      average_accuracy = average_accuracy,
                                      num_x = num_x,
                                      num_y = num_y,
                                      num_z = num_z})

              filename_0 = "/run/user/1000/gvfs/smb-share:server=hpc-fs.qut.edu.au,share=n9314181/control_dataset_off_vel"
              filename_0 = filename_0 .. average_accuracy .. "_" .. k
              torch.save(filename_0 .. ".t7", {object_pose = target_pose_,
                                      end_effector_pose = target_pose_e,
                                      desired_pose = desired_pose_,
                                      init_pose = initial_arm_pose_,
                                      arm_pose = current_arm_pose_,
                                      low_dim_s = low_dim_s_,
                                      low_dim_s_e = low_dim_s_e,
                                      label = label_v_,
                                      vel_cmd = vel_cmd_,
                                      sample_amount = #low_dim_s_})

              filename_0 = "/run/user/1000/gvfs/smb-share:server=hpc-fs.qut.edu.au,share=n9314181/no_config_cases_off_vel"
              filename_0 = filename_0 .. average_accuracy .. "_" .. k
              torch.save(filename_0 .. ".t7", {target_pose = no_config_target,
                                      low_dim_s = no_config_low_dim_s,
                                      label = no_config_vel,
                                      vel_cmd = no_config_vel,
                                      sample_amount = no_config_index})

          end
        end

      end
    end


    -- Get the dimensionality of each element in the dataset
    -- local image_dim = image_[1]:size()
    local label_dim = #label_[1]
    -- Save the dataset to a t7 file
    local filename_ = "/run/user/1000/gvfs/smb-share:server=hpc-fs.qut.edu.au,share=n9314181/case_dataset_off_vel"
    filename_ = filename_ .. average_accuracy .. "_" .. k
    torch.save(filename_ .. ".t7", {object_pose = target_pose_,
                            end_effector_pose = target_pose_e,
                            desired_pose = desired_pose_,
                            init_pose = initial_arm_pose_,
                            sample_amount = sample_index,
                            average_accuracy = average_accuracy,
                            num_x = num_x,
                            num_y = num_y,
                            num_z = num_z})
    print('Case dataset generated:', filename_ .. '.t7')

    filename_ = "/run/user/1000/gvfs/smb-share:server=hpc-fs.qut.edu.au,share=n9314181/control_dataset_off_vel"
    filename_ = filename_ .. average_accuracy .. "_" .. k
    torch.save(filename_ .. ".t7", {object_pose = target_pose_,
                            end_effector_pose = target_pose_e,
                            desired_pose = desired_pose_,
                            init_pose = initial_arm_pose_,
                            arm_pose = current_arm_pose_,
                            low_dim_s = low_dim_s_,
                            low_dim_s_e = low_dim_s_e,
                            label = label_v_,
                            vel_cmd = vel_cmd_,
                            sample_amount = #low_dim_s_})
    print('Control dataset generated:', filename_ .. '.t7')

    filename_ = "/run/user/1000/gvfs/smb-share:server=hpc-fs.qut.edu.au,share=n9314181/no_config_cases_off_vel"
    filename_ = filename_ .. average_accuracy .. "_" .. k
    torch.save(filename_ .. ".t7", {target_pose = no_config_target,
                            low_dim_s = no_config_low_dim_s,
                            label = no_config_vel,
                            vel_cmd = no_config_vel,
                            sample_amount = no_config_index})

    print('No config case dataset generated:', filename_ .. '.t7')
    print("Dataset Property: \n")
    -- print("image_dim: \n", image_dim)
    print("label_dim: \n", label_dim)
    print("case amount: ", sample_index)
    print("vel simple amount: ", #low_dim_s_)
    print("no config case amount: ", no_config_index)
    print("accuracy: ", average_accuracy)
    print("num_x: ", num_x)
    print("num_y: ", num_y)
    print("num_z: ", num_z)
    io.flush()
    collectgarbage()
  end

end

function dctrl:generate_images_for_all_ctrl_datasets(name, start_index, end_index)
  for i=start_index,end_index do
    ctrl_dataset = "/run/user/1000/gvfs/smb-share:server=hpc-fs.qut.edu.au,share=n9314181/" .. name .. i .. '.t7'
    self:generate_images_for_control_dataset(ctrl_dataset, name .. i)
    io.flush()
    collectgarbage()
  end
  print("All Control Datasets Finished!!!")
end


-- Generate a dataset for perception
-- TODO: save the images in the trajectory manner, each trajectory in a t7 file
function dctrl:generate_images_for_control_dataset(ctrl_dataset_name, name)
  -- Load control dataset
  local ctrl_dataset = self:load_pre_constructed_dataset(ctrl_dataset_name)
  print("Dataset Loaded: ", ctrl_dataset_name)

  local low_dim_s_e = ctrl_dataset.low_dim_s_e
  local label = ctrl_dataset.label
  local arm_pose = ctrl_dataset.arm_pose
  local sample_amount = ctrl_dataset.sample_amount
  print("Samples to be generated: ", sample_amount)
  print("Generating images ...... ")

  local image300 = {}
  local image400 = {}
  local object_position = {}
  local arm_pose_ = {}
  local vel = {}
  local low_dim_s_e_ = {}
  local curr_index = 0
  local t7_saved = 0
  for i=1,sample_amount do
    local target = self:recoverLowDimTensor(low_dim_s_e[i]:narrow(1,8,3))
    target = target:totable()
    target[3] = target[3] - 0.0325
    local target_orien = self:randTargetOrientation()
    self.vrep:setObjectOrientation(self.cuboid, self.robot_base, target_orien)
    self.vrep:setObjectPosition(self.cuboid, self.robot_base, target)
    self:setCameraPose(true)
    self:randTablePose(true)
    local table_position = self.vrep:getObjectPosition(self.table, self.robot_base)
    -- print("Table Position: ", torch.Tensor(table_position))
    if table_position[3] > target[3] - 0.0825 then
      table_position[3] = target[3] - 0.0825
      -- print("Table position calibrated!!!")
    end
    self.vrep:setObjectPosition(self.table, self.robot_base, table_position)
    self:randClutters()
    -- local pose = self:initArmPose()
    self:setArmTargetPose('left',arm_pose[i])
    for i=1,100 do
      if torch.Tensor(self:getArmVelocity('left')):abs():sum() < 0.001 then
        -- print(i)
        completion = true
        break
      elseif i==100 then
        print("Reaching failed, filter out!!!")
      end
    end

    -- local position = self.vrep:getObjectPosition(self.cuboid, self.robot_base)
    -- local position_e = self.vrep:getObjectPosition(self.cuboid, self.robot_base)
    -- local position_t2e = self.vrep:getObjectPosition(self.cuboid, self.robot_base)


    if self.image_format == 'RGB' then
      im = self.vrep:getRGBImage(self.right_camera)
    else
      im = self.vrep:getGreyImage(self.right_camera)
    end
    local cropped_image300 = image.crop(im,120,20,420,320)
    local cropped_image400 = image.crop(im,0,0,400,400)
    -- print("Image size: ", cropped_image:size())

    -- Add into the dataset
    curr_index = curr_index + 1
    image300[curr_index] = cropped_image300:float():div(255)
    image400[curr_index] = cropped_image400:float():div(255)
    object_position[curr_index] = table.copy(target)
    arm_pose_[curr_index] = table.copy(arm_pose[i])
    vel[curr_index] = table.copy(label[i])
    low_dim_s_e_[curr_index] = low_dim_s_e[i]:clone()

    -- Visualize images and output results
    -- if display then
    win_input1 = image.display{image=image300[curr_index], win=win_input1}
    win_input2 = image.display{image=image400[curr_index], win=win_input2}
    -- end
    -- print("=====================================")
    -- print("Sample Index: ", curr_index)
    -- print("Object Position: ", torch.Tensor(object_position[curr_index]))
    -- print("Arm Pose: ", torch.Tensor(arm_pose_[curr_index]))
    -- print("Vel: ", torch.Tensor(vel[curr_index]))
    -- print("Low Dim S: ", low_dim_s_e_[curr_index])

    if curr_index%50 == 0 then
        print("Samples Collected: ", curr_index)
    end

    -- Save the dataset to a t7 file
    if curr_index == 3000 or i == sample_amount then
      t7_saved = t7_saved + 1
      local filename_ = "/run/user/1000/gvfs/smb-share:server=hpc-fs.qut.edu.au,share=n9314181/image300_" .. name .. "_" .. t7_saved
      torch.save(filename_ .. ".t7", {image300 = image300,
                              object_position = object_position,
                              arm_pose = arm_pose_,
                              vel = vel,
                              low_dim_s_e = low_dim_s_e_,
                              sample_amount = curr_index})
      print('Image dataset generated:', filename_ .. '.t7')

      filename_ = "/run/user/1000/gvfs/smb-share:server=hpc-fs.qut.edu.au,share=n9314181/image400_" .. name .. "_" .. t7_saved
      torch.save(filename_ .. ".t7", {image400 = image400,
                              object_position = object_position,
                              arm_pose = arm_pose_,
                              vel = vel,
                              low_dim_s_e = low_dim_s_e_,
                              sample_amount = curr_index})
      print('Image dataset generated:', filename_ .. '.t7')
      print("Samples Collected: ", curr_index)

      image300 = nil
      image400 = nil
      object_position = nil
      arm_pose_ = nil
      vel = nil
      low_dim_s_e_ = nil

      collectgarbage()

      image300 = {}
      image400 = {}
      object_position = {}
      arm_pose_ = {}
      vel = {}
      low_dim_s_e_ = {}
      curr_index = 0
    end
  end

  print("One Control Dataset Finished!!!")

end

-- Generate a dataset for perception
function dctrl:generate_test_set(display)

  -- local self_collision_allowance = {0.88, 0.88, 0.96}
  -- local pos_min = {0.1, -0.2625, 0.025}
  -- local pos_max = {0.7, 0.2625, 0.025}
  local pos_min = table.copy(self.target_min)
  local pos_max = table.copy(self.target_max)
  local average_accuracy = 0.1
  -- local average_accuracy = 0.2
  -- local rand_scalar = 2 * average_accuracy

  self:setCameraPose(false)

  print("Generating testing samples ...... ")

  local target = table.copy(self.target_min)

  local num_x = math.floor((pos_max[1] - pos_min[1])/average_accuracy + 0.5)
  local num_y = math.floor((pos_max[2] - pos_min[2])/average_accuracy + 0.5)
  print("num_x: ", num_x)
  print("num_y: ", num_y)

  local image300 = {}
  local image400 = {}
  local object_position = {}
  local target_arm_pose = {}
  local arm_pose = {}
  local sample_index = 0

  for i=1,num_x do
    for j=1,num_y do
      target[1] = pos_min[1] + (i-1 + torch.rand(1)[1]) * average_accuracy
      if target[1] > pos_max[1] then
        target[1] = pos_max[1]
      end
      target[2] = pos_min[2] + (j-1 + torch.rand(1)[1]) * average_accuracy
      if target[2] > pos_max[2] then
        target[2] = pos_max[2]
      end
      local init_pose = self:initArmPose()
      local target_orien = self:randTargetOrientation()
      self.vrep:setObjectOrientation(self.cuboid, self.robot_base, target_orien)
      self.vrep:setObjectPosition(self.cuboid, self.robot_base, target)

      local desired_pose = self.vrep:getDesiredConfig(0.2)
      if desired_pose == nil then
        print("No desired config, filter out !!!!")
      elseif #desired_pose == 0 then
        print("No desired config, filter out !!!!")
      else
        -- print("*******",i)
        self:setArmTargetPose('left',desired_pose)
        for i=1,100 do
          if torch.Tensor(self:getArmVelocity('left')):abs():sum() < 0.001 then
            break
          elseif i==100 then
            print("Reaching failed, filter out!!!")
          end
        end
        local position = self.vrep:getObjectPosition(self.left_gripper, self.robot_base)
        position[3] = position[3] - self.target_h
        self.vrep:setObjectPosition(self.cuboid, self.robot_base, position)
        local pose = self:generateRandPose(init_pose, desired_pose)
        local collision, reached = self:detectCollisions()
        while collision and not reached do
          print("Initial collision happened, filter out !!!!")
          pose = self:generateRandPose(init_pose, desired_pose)
          collision, reached = self:detectCollisions()
        end

        self:randClutters()

        local im
        if self.image_format == 'RGB' then
          im = self.vrep:getRGBImage(self.right_camera)
        else
          im = self.vrep:getGreyImage(self.right_camera)
        end
        -- TODO develop codes to crop and resize the image
        local cropped_image300 = image.crop(im,120,20,420,320)
        local cropped_image400 = image.crop(im,0,0,400,400)

        -- Add into the dataset
        sample_index = sample_index + 1
        -- image_[sample_index] = cropped_image:float():div(255)
        -- print(image_[sample_index]:max())
        image300[sample_index] = cropped_image300:float():div(255)
        image400[sample_index] = cropped_image400:float():div(255)
        object_position[sample_index] = table.copy(position)
        target_arm_pose[sample_index] = table.copy(desired_pose)
        arm_pose[sample_index] = table.copy(pose)

        -- Visualize images and output results
        -- if display then
        win_input1 = image.display{image=image300[sample_index], win=win_input1}
        win_input2 = image.display{image=image400[sample_index], win=win_input2}
        -- end

        print("=====================================")
        print("Sample Index: ", sample_index)
        print("Object Position: ", torch.Tensor(object_position[sample_index]))
        print("Target Arm Pose: ", torch.Tensor(target_arm_pose[sample_index]))
        print("Arm Pose: ", torch.Tensor(arm_pose[sample_index]))


        -- Save the dataset to a t7 file
        local filename_ = "image300_test_set_"
        filename_ = filename_ .. average_accuracy
        torch.save(filename_ .. ".t7", {image = image300,
                                object_position = object_position,
                                target_arm_pose = target_arm_pose,
                                arm_pose = arm_pose,
                                sample_amount = sample_index})
        print('Test dataset saved:', filename_ .. '.t7')

        filename_ = "image400_test_set_"
        filename_ = filename_ .. average_accuracy
        torch.save(filename_ .. ".t7", {image = image400,
                                object_position = object_position,
                                target_arm_pose = target_arm_pose,
                                arm_pose = arm_pose,
                                sample_amount = sample_index})
        print('Test dataset saved:', filename_ .. '.t7')

      end

    end
  end
  print("Testset Genration Done")
  print("Sample amount: ", sample_index)
  print("accuracy: ", average_accuracy)
  print("num_x: ", num_x)
  print("num_y: ", num_y)
  io.flush()
  collectgarbage()

end

-- Generate a dataset for perception with domain randomization
-- TODO: to finish
function dctrl:generate_test_set_dr(display)

  -- local self_collision_allowance = {0.88, 0.88, 0.96}
  -- local pos_min = {0.1, -0.2625, 0.025}
  -- local pos_max = {0.7, 0.2625, 0.025}
  local pos_min = table.copy(self.target_min)
  local pos_max = table.copy(self.target_max)
  local average_accuracy = 0.1
  -- local average_accuracy = 0.2
  -- local rand_scalar = 2 * average_accuracy

  self:setCameraPose(false)

  print("Generating testing samples ...... ")

  local target = table.copy(self.target_min)

  local num_x = math.floor((pos_max[1] - pos_min[1])/average_accuracy + 0.5)
  local num_y = math.floor((pos_max[2] - pos_min[2])/average_accuracy + 0.5)
  print("num_x: ", num_x)
  print("num_y: ", num_y)

  local image300 = {}
  local image400 = {}
  local object_position = {}
  local target_arm_pose = {}
  local arm_pose = {}
  local sample_index = 0

  for i=1,num_x do
    for j=1,num_y do
      target[1] = pos_min[1] + (i-1 + torch.rand(1)[1]) * average_accuracy
      if target[1] > pos_max[1] then
        target[1] = pos_max[1]
      end
      target[2] = pos_min[2] + (j-1 + torch.rand(1)[1]) * average_accuracy
      if target[2] > pos_max[2] then
        target[2] = pos_max[2]
      end
      local init_pose = self:initArmPose()
      local target_orien = self:randTargetOrientation()
      self.vrep:setObjectOrientation(self.cuboid, self.robot_base, target_orien)
      self.vrep:setObjectPosition(self.cuboid, self.robot_base, target)

      local desired_pose = self.vrep:getDesiredConfig(0.2)
      if desired_pose == nil then
        print("No desired config, filter out !!!!")
      elseif #desired_pose == 0 then
        print("No desired config, filter out !!!!")
      else
        -- print("*******",i)
        self:setArmTargetPose('left',desired_pose)
        for i=1,100 do
          if torch.Tensor(self:getArmVelocity('left')):abs():sum() < 0.001 then
            break
          elseif i==100 then
            print("Reaching failed, filter out!!!")
          end
        end
        local position = self.vrep:getObjectPosition(self.left_gripper, self.robot_base)
        position[3] = position[3] - self.target_h
        self.vrep:setObjectPosition(self.cuboid, self.robot_base, position)
        local pose = self:generateRandPose(init_pose, desired_pose)
        local collision, reached = self:detectCollisions()
        while collision and not reached do
          print("Initial collision happened, filter out !!!!")
          pose = self:generateRandPose(init_pose, desired_pose)
          collision, reached = self:detectCollisions()
        end

        self:randClutters()

        local im
        if self.image_format == 'RGB' then
          im = self.vrep:getRGBImage(self.right_camera)
        else
          im = self.vrep:getGreyImage(self.right_camera)
        end
        -- TODO develop codes to crop and resize the image
        local cropped_image300 = image.crop(im,120,20,420,320)
        local cropped_image400 = image.crop(im,0,0,400,400)

        -- Add into the dataset
        sample_index = sample_index + 1
        -- image_[sample_index] = cropped_image:float():div(255)
        -- print(image_[sample_index]:max())
        image300[sample_index] = cropped_image300:float():div(255)
        image400[sample_index] = cropped_image400:float():div(255)
        object_position[sample_index] = table.copy(position)
        target_arm_pose[sample_index] = table.copy(desired_pose)
        arm_pose[sample_index] = table.copy(pose)

        -- Visualize images and output results
        -- if display then
        win_input1 = image.display{image=image300[sample_index], win=win_input1}
        win_input2 = image.display{image=image400[sample_index], win=win_input2}
        -- end

        print("=====================================")
        print("Sample Index: ", sample_index)
        print("Object Position: ", torch.Tensor(object_position[sample_index]))
        print("Target Arm Pose: ", torch.Tensor(target_arm_pose[sample_index]))
        print("Arm Pose: ", torch.Tensor(arm_pose[sample_index]))


        -- Save the dataset to a t7 file
        local filename_ = "image300_test_set_"
        filename_ = filename_ .. average_accuracy
        torch.save(filename_ .. ".t7", {image = image300,
                                object_position = object_position,
                                target_arm_pose = target_arm_pose,
                                arm_pose = arm_pose,
                                sample_amount = sample_index})
        print('Test dataset saved:', filename_ .. '.t7')

        filename_ = "image400_test_set_"
        filename_ = filename_ .. average_accuracy
        torch.save(filename_ .. ".t7", {image = image400,
                                object_position = object_position,
                                target_arm_pose = target_arm_pose,
                                arm_pose = arm_pose,
                                sample_amount = sample_index})
        print('Test dataset saved:', filename_ .. '.t7')

      end

    end
  end
  print("Testset Genration Done")
  print("Sample amount: ", sample_index)
  print("accuracy: ", average_accuracy)
  print("num_x: ", num_x)
  print("num_y: ", num_y)
  io.flush()
  collectgarbage()

end


function dctrl:sort_real_dataset()

  local data400 = self:load_pre_constructed_dataset('image400_real_data_50_3.t7')
  local data300 = self:load_pre_constructed_dataset('image300_real_data_50_3.t7')

  local data_amount = data300.sample_amount

  -- local omit_targets = {26}
  -- local omit_targets = {25,26,27}
  -- local omit_targets = {70,71,72,85,86,87,94,95,96}
  -- local omit_targets = {124,125,126}
  local omit_index = 1
  local image400 = {}
  local image300 = {}
  local target_pose = {}
  local arm_pose = {}
  local low_dim_s_e = {}
  local vel = {}
  local sample_index = 0


  for i=1,data_amount do
    if i == omit_targets[omit_index] then
      print("One sample filtterd out: ", i)
      if omit_index < #omit_targets then
        omit_index = omit_index + 1
      end
    else
      self:setArmTargetPose('left',data300.target_arm_pose[i])
      for j=1,100 do
        if torch.Tensor(self:getArmVelocity('left')):abs():sum() < 0.001 then
          local position_e = self.vrep:getObjectPosition(self.left_gripper, self.robot_base)
          -- -- Compensate the suction gripper height differences
          -- position_e[3] = position_e[3] - 0.005
          local current_pos = table.copy(data300.arm_pose[i])
          local low_dim_features = self:outputLowDimTensor(current_pos, position_e)
          position_e[3] = position_e[3] - 0.0325
          self.vrep:setObjectPosition(self.cuboid, self.robot_base, position_e)
          local desired_pose = self.vrep:getDesiredConfig(0.2)
          if desired_pose == nil then
            print("No desired config, filter out!!!")
          elseif #desired_pose == 0 then
            print("No desired config, filter out!!!")
          else
            local vel_ = {}
            for i=1,self.n_joint do
              vel_[i] = desired_pose[i] - current_pos[i]
              if vel_[i] > 1.0 then
                vel_[i] = 1.0
              elseif vel_[i] < -1.0 then
                vel_[i] = -1.0
              end
            end
            local vel_sum = torch.Tensor(vel_):abs():sum()
            -- print("vel sum: ", vel_sum)
            -- if vel_sum < 0.01 then
            if vel_sum < 0.01 then
              vel_ = {0.0,0.0,0.0,0.0,0.0,0.0,0.0}
            end

            sample_index = sample_index + 1
            image300[sample_index] = data300.image[i]:clone()
            image400[sample_index] = data400.image[i]:clone()
            target_pose[sample_index] = table.copy(position_e)
            arm_pose[sample_index] = table.copy(data300.arm_pose[i])
            low_dim_s_e[sample_index] = low_dim_features
            vel[sample_index] = table.copy(vel_)

            print("=====================================")
            print("Sample Index: ", sample_index)
            print("Traget Arm Pose: ", torch.Tensor(target_pose[sample_index]))
            print("Current Arm Pose: ", torch.Tensor(arm_pose[sample_index]))
            print("Low Dim s: ", low_dim_s_e[sample_index])
            print("Vel: ", torch.Tensor(vel[sample_index]))
            win_input1 = image.display{image=image400[sample_index], win=win_input1}
            win_input2 = image.display{image=image300[sample_index], win=win_input2}
          end


          break
        elseif j==100 then
          print("Reaching failed, filter out!!!")
        end
      end


      -- print("Label V: ", torch.Tensor(label_v_[#label_v_]))

      -- Save the dataset to a t7 file
      local filename_ = "sorted_image300_real_data_50_3"
      torch.save(filename_ .. ".t7", {image = image300,
                              object_position = target_pose,
                              arm_pose = arm_pose,
                              low_dim_s_e = low_dim_s_e,
                              vel = vel,
                              sample_amount = sample_index})
      print('Real dataset saved:', filename_ .. '.t7')
      filename_ = "sorted_image400_real_data_50_3"
      torch.save(filename_ .. ".t7", {image = image400,
                              object_position = target_pose,
                              arm_pose = arm_pose,
                              low_dim_s_e = low_dim_s_e,
                              vel = vel,
                              sample_amount = sample_index})
      print('Real dataset saved:', filename_ .. '.t7')

    end
  end
  io.flush()
  collectgarbage()

  print("Data Sorted Out!!! Go for a break!!!")

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

function dctrl:load_pre_constructed_dataset(file)
    -- try to load the dataset
    local err_msg,  dataset= pcall(torch.load, file)
    if not err_msg then
        error("Could not find the dataset file ")
    end
    return dataset
end
