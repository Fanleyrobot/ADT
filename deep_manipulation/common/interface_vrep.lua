--[[ File
@description:
    This class contains wrapped interfaces to VREP.
@version: V0.00
@author: Fangyi Zhang   email:gzzhangfangyi@gmail.com
@acknowledgement:
    ARC Centre of Excellence for Robotic Vision (ACRV)
    Queensland Univsersity of Technology (QUT)
@history:
    V0.00   22/04/2017  developed the first version
]]

require 'torch'
require 'image'
require 'remoteApiLua'

-- construct a class
local vrep = torch.class('interface_vrep')

-- Initialize the vrep environment and connection
function vrep:__init(args)
    -- vrep setup
    self.vrep_app = args.vrep_app
    self.vrep_scene = args.vrep_scene

    self.debug_mode = args.debug_mode or false -- debugging or not, when debugging, no headless vrep will be launched
    self.syn_mode = args.syn_mode

    self.port_num = args.prot_num or 29999
    self.waitUntilConn = args.waitUntilConn or true
    self.doNotRecon = args.doNotRecon or true
    self.connTimeout = args.connTimeout or -10000 -- time out after 10000 ms for blocking function calls, but 5000ms for the first connection
    print("connTimeout: ", self.connTimeout)
    self.commCycle = args.commCycle or 5 -- communication cycle in ms, i.e., 5 ms

    self.vrep_launch_wait = args.vrep_launch_wait or 10 -- 10s
    self.stream_init_wait = args.stream_init_wait or 1 -- 0.01s

    simxFinish(-1) -- just in case, close all opened connections
    if not self.debug_mode then
      assert(self.vrep_app, "Please provide the path to vrep.sh!!!")
      assert(self.vrep_scene, "Please provide the path to a vrep scene file (e.g., Baxter_Object_Picking.ttt)!!!")
      self:close_vrep() -- just in case, close existing vrep processes
      self:run_vrep(self.vrep_app,self.vrep_scene) -- Launch Vrep
    end

    -- Try to connect to vrep for maximum 3 times
    for i=1,3 do
      self.clientID=simxStart('127.0.0.1',self.port_num,self.waitUntilConn,self.doNotRecon,self.connTimeout,self.commCycle)
      if clientID~=-1 then
          print('Connected to remote API server in Trial: ', i)
          -- enable/disable the synchronous mode on the client:
          simxSynchronous(self.clientID, self.syn_mode)
          print('Synchronous Mode: ', self.syn_mode)
          if self.syn_mode then
            print('VREP is now running in synchronous mode.')
            print('Please remember to call stepsForward(n) to make simulation n steps forward!')
          end

          break
      else
          print('Failed connecting to remote API server')
      end
    end

end

-- Launch a headless vrep process
function vrep:run_vrep(vrep, scene)
  os.execute(vrep .. ' -h -s ' .. scene .. ' &')
  os.execute("sleep " .. self.vrep_launch_wait)
  -- TODO set a trigger parameter to tell the remote terminal vrep terminal is ready
end

-- Close vrep processes
function vrep:close_vrep()
  if self.clientID then
    -- Before closing the connection to V-REP, make sure that the last command sent out had time to arrive. You can guarantee this with (for example):
    simxGetPingTime(self.clientID)

    -- Now close the connection to V-REP:
    simxFinish(self.clientID)
  end

  -- Close the vrep process
  local vrep_id = io.popen('pidof vrep'):read('*a')
  if vrep_id ~= "" then
    os.execute('kill -9 ' .. vrep_id)
  end
  print('Vrep ended')
end

-- Start simulation
function vrep:startSimulation()
  local rs = simxStartSimulation(self.clientID,simx_opmode_blocking)
  if rs ~= simx_return_ok then
    print('Failed to start simulation!!! Error Code: ', rs)
  end
end

-- Stop simulation
function vrep:stopSimulation()
  local rs = simxStopSimulation(self.clientID,simx_opmode_blocking)
  if rs ~= simx_return_ok then
    print('Failed to stop simulation!!! Error Code: ', rs)
  end
end

-- One/multiple simulation step forward for synchronous mode
function vrep:stepsForward(steps)
  local n = steps or 1
  for i=1,n do
    simxSynchronousTrigger(self.clientID)
  end
end

-- Get object handle
function vrep:getObjectHandle(name)
  local rs, object_handle = simxGetObjectHandle(self.clientID, name, simx_opmode_blocking)
  if rs ~= simx_return_ok then
    print('Get Object Handle Failed!!! Object Name: ', name)
  end
  return object_handle
end

-- Initialize an RGB camera to stream
function vrep:initRGBCamera(camera_handle)
  local rs, im, size = simxGetVisionSensorImage(self.clientID, camera_handle, 0, simx_opmode_streaming)
  os.execute('sleep ' .. self.stream_init_wait) -- wait for the buffer to be filled
  rs, im, size = simxGetVisionSensorImage(self.clientID, camera_handle, 0, simx_opmode_buffer)
  if rs == simx_return_ok then
    print('RGB Camera Initialized. Resolution: ', torch.Tensor(size))
  else
    print('RGB Camera Initialization Failed!!! Error Code: ', rs)
  end
  return rs
end

-- Initialize an Grey-scale camera to stream
function vrep:initGreyCamera(camera_handle)
  local rs, im, size = simxGetVisionSensorImage(self.clientID, camera_handle, 1, simx_opmode_streaming)
  os.execute('sleep ' .. self.stream_init_wait) -- wait for the buffer to be filled
  rs, im, size = simxGetVisionSensorImage(self.clientID, camera_handle, 1, simx_opmode_buffer)
  if rs == simx_return_ok then
    print('Grey-scale Camera Initialized. Resolution: ', torch.Tensor(size))
  else
    print('Grey-scale Camera Initialization Failed!!! Error Code: ', rs)
  end
  return rs
end

-- Initialize a depth camera to stream
function vrep:initDepthCamera(camera_handle)
  local rs, im, size = simxGetVisionSensorDepthBuffer(self.clientID, camera_handle, simx_opmode_streaming)
  os.execute('sleep ' .. self.stream_init_wait) -- wait for the buffer to be filled
  rs, im, size = simxGetVisionSensorDepthBuffer(self.clientID, camera_handle, simx_opmode_buffer)
  if rs == simx_return_ok then
    print('Depth Camera Initialized. Resolution: ', torch.Tensor(size))
  else
    print('Depth Camera Initialization Failed!!! Error Code: ', rs)
  end
  return rs
end

-- Get RGB image
function vrep:getRGBImage(camera_handle)
  local rs, im, size = simxGetVisionSensorImage(self.clientID, camera_handle, 0, simx_opmode_buffer)
  if rs == simx_return_ok then
    im = self:stringI2torchTensor(im, size)
    return im
  else
    print('Failed to get RGB image!!! Error Code: ', rs)
  end
end

-- Get grey-scale image
function vrep:getGreyImage(camera_handle)
  local rs, im, size = simxGetVisionSensorImage(self.clientID, camera_handle, 1, simx_opmode_buffer)
  if rs == simx_return_ok then
    im = self:stringI2torchTensor(im, size)
    return im
  else
    print('Failed to get grey-scale image!!! Error Code: ', rs)
  end
end

-- Get depth image
function vrep:getDepthImage(camera_handle)
  local rs, im, size = simxGetVisionSensorDepthBuffer(self.clientID, camera_handle, simx_opmode_buffer)
  if rs == simx_return_ok then
    im = self:stringI2torchTensor(im, size)
    return im
  else
    print('Failed to get depth image!!! Error Code: ', rs)
  end
end

-- Convert a string image received from VREP to a tensor
function vrep:stringI2torchTensor(i, res)
  local n_pixel = res[1] * res[2]
  local s_len = string.len(i)

  -- Convert a string image to a tensor
  local im = torch.ByteTensor(s_len)
  for k=1,s_len do
    im[k] = string.byte(i,k)
  end

  if s_len == 3 * n_pixel then -- RGB image
    im = im:resize(res[2],res[1],3)
    local im_temp = torch.ByteTensor(3,res[2],res[1])
    for j=1,3 do
      im_temp:select(1,j):copy(im:select(3,j))
    end
    im = im_temp
  else -- Grey-scale image
    im = im:resize(res[2],res[1])
  end

  im = image.vflip(im)

  return im

end

-- Get joint position
function vrep:getJointPosition(joint_handle)
  local rs, joint_position = simxGetJointPosition(self.clientID, joint_handle, simx_opmode_blocking)
  if rs ~= simx_return_ok then
    print('Failed to get joint position!!! Error Code: ', rs)
  end
  return joint_position
end

-- Set joint position
function vrep:setJointPosition(joint_handle, pos)
  local rs = simxSetJointTargetPosition(self.clientID, joint_handle, pos, simx_opmode_oneshot)
  if rs == -1 then
    print('Failed to set joint position!!! Error Code: ', rs)
  end
  return rs
end

-- Get joint velocity
function vrep:getJointVelocity(joint_handle)
  local rs, joint_vel = simxGetObjectFloatParameter(self.clientID, joint_handle, sim_jointfloatparam_velocity, simx_opmode_blocking)
  if rs ~= simx_return_ok then
    print('Failed to get joint velocity!!! Error Code: ', rs)
  end
  return joint_vel
end

-- Set joint velocity
function vrep:setJointVelocity(joint_handle, vel)
  local rs = simxSetJointTargetVelocity(self.clientID, joint_handle, vel, simx_opmode_oneshot)
  if rs == -1 then
    print('Failed to set joint velocity!!! Error Code: ', rs)
  end
  return rs
end

-- TODO Add the functions for force

-- Get object position
function vrep:getObjectPosition(object, relative_object)
  local rs, pos = simxGetObjectPosition(self.clientID, object, relative_object, simx_opmode_blocking)
  if rs ~= simx_return_ok then
    print('Failed to get object position!!! Error Code: ', rs)
  end
  return pos
end

-- Get object orientation
function vrep:getObjectOrientation(object, relative_object)
  local rs, eu_angle = simxGetObjectOrientation(self.clientID, object, relative_object, simx_opmode_blocking)
  if rs ~= simx_return_ok then
    print('Failed to get object orientation!!! Error Code: ', rs)
  end
  return eu_angle
end

-- Set object position
function vrep:setObjectPosition(object, relative_object, pos)
  local rs = simxSetObjectPosition(self.clientID, object, relative_object, pos, simx_opmode_oneshot)
  if rs == -1 then
    print('Set Object Position Failed!!! Error Code: ', rs)
  end
  return rs
end

-- Set object orientation
function vrep:setObjectOrientation(object, relative_object, eu_angle)
  local rs = simxSetObjectOrientation(self.clientID, object, relative_object, eu_angle, simx_opmode_oneshot)
  if rs == -1 then
    print('Set Object Orientation Failed!!! Error Code: ', rs)
  end
  return rs
end

-- Get collision handle
function vrep:getCollisionHandle(name)
  local rs, collision_handle = simxGetCollisionHandle(self.clientID, name, simx_opmode_blocking)
  if rs ~= simx_return_ok then
    print('Failed to get collision handle!!! Collision Name: ', name)
  end
  return collision_handle
end

-- Initialize collision detect
function vrep:initCollisionDetect(collision_handle)
  local rs, state = simxReadCollision(self.clientID, collision_handle, simx_opmode_streaming)
  os.execute('sleep ' .. self.stream_init_wait) -- wait for the buffer to be filled
  rs, state = simxReadCollision(self.clientID, collision_handle, simx_opmode_buffer)
  if rs == simx_return_ok then
    print('Collision Detection Streaming Initialized. Collision Handle: ', collision_handle)
  else
    print('Collision Detection Streaming Initialization Failed!!! Error Code: ', rs)
  end
  return rs
end

-- Detect collision
function vrep:detectCollision(collision_handle)
  local rs, state = simxReadCollision(self.clientID, collision_handle, simx_opmode_buffer)
  if rs ~= simx_return_ok then
    print('Failed to detect collision!!! Error Code: ', rs)
  end
  return state
end

-- Get distance handle
function vrep:getDistanceHandle(name)
  local rs, distance_handle = simxGetDistanceHandle(self.clientID, name, simx_opmode_blocking)
  if rs ~= simx_return_ok then
    print('Failed to get distance handle!!! Distance Name: ', name)
  end
  return distance_handle
end

-- Initialize distance detect
function vrep:initDistanceDetect(distance_handle)
  local rs, distance = simxReadDistance(self.clientID, distance_handle, simx_opmode_streaming)
  os.execute('sleep ' .. self.stream_init_wait) -- wait for the buffer to be filled
  rs, distance = simxReadDistance(self.clientID, distance_handle, simx_opmode_buffer)
  if rs == simx_return_ok then
    print('Distance Detection Streaming Initialized. Distance Handle: ', distance_handle)
  else
    print('Distance Detection Streaming Initialization Failed!!! Error Code: ', rs)
  end
  return rs
end

-- Detect distance
function vrep:detectDistance(distance_handle)
  local rs, distance = simxReadDistance(self.clientID, distance_handle, simx_opmode_buffer)
  if rs ~= simx_return_ok then
    print('Failed to detect distance!!! Error Code: ', rs)
  end
  return distance
end

-- Get simulation time
function vrep:getSimTime()
  local sim_time = simxGetLastCmdTime(self.clientID)
  return sim_time
end

-- Enable/Disable vision sensor rendering
function vrep:setImageRendering(enable)
  local rs = simxSetBooleanParameter(self.clientID, sim_boolparam_vision_sensor_handling_enabled, enable, simx_opmode_blocking)
  if rs ~= simx_return_ok then
    print('Failed to set vision sensor rendering state!!! Error Code: ', rs)
  else
    print("Vision Sensors Enabled: ", enable)
  end
  return rs
end

-- Get desired config for a target
function vrep:getDesiredConfig(search_distanceTh, search_timeTh, search_trials)
  search_distanceTh = search_distanceTh or 0.65
  -- print("Distance Threshold: ", search_distanceTh)
  search_timeTh = search_timeTh or 20
  search_trials = search_trials or 3
  local rs,outInts,jp,outStrings,outBuffer=simxCallScriptFunction(self.clientID, "Baxter_leftArm_joint1", sim_scripttype_childscript, "getDesiredConfig", {search_timeTh, search_trials}, {search_distanceTh}, {}, "", simx_opmode_blocking)
  if rs ~= simx_return_ok then
    print('Failed to receive desired config!!! Error Code: ', rs)
  elseif #jp==0 then
    print('Failed to find an available config for the current target!!!')
  else
    print("Desired Config: ", torch.Tensor(jp))
  end
  return jp
end

-- Enable position control
function vrep:enableJointPositionCtrl()
  local rs,outInts,outFloats,outStrings,outBuffer=simxCallScriptFunction(self.clientID, "Baxter_leftArm_joint1", sim_scripttype_childscript, "enableJointPositionCtrl", {}, {}, {}, "", simx_opmode_blocking)
  if rs ~= simx_return_ok then
    print('Failed to enable joint position control!!! Error Code: ', rs)
  elseif outInts[1] == 1 then
    rs = 1
    print("Joint Position Control Enabled")
  end
  return rs
end

-- Enable velocity control
function vrep:enableJointVelocityCtrl()
  local rs,outInts,outFloats,outStrings,outBuffer=simxCallScriptFunction(self.clientID, "Baxter_leftArm_joint1", sim_scripttype_childscript, "enableJointVelocityCtrl", {}, {}, {}, "", simx_opmode_blocking)
  if rs ~= simx_return_ok then
    print('Failed to enable joint velocity control!!! Error Code: ', rs)
  elseif outInts[1] == 1 then
    rs = 1
    print("Joint Velocity Control Enabled")
  end
  return rs
end

-- -- Set object size
-- function vrep:scaleObjectSize(object_handle, scalar)
--   local rs = simxSetObjectFloatParameter(self.clientID, object_handle, sim_shapefloatparam_texture_scaling_x ,scalar[1], simx_opmode_oneshot)
--   rs = simxSetObjectFloatParameter(self.clientID, object_handle, sim_shapefloatparam_texture_scaling_y, scalar[2], simx_opmode_oneshot)
--   if rs ~= simx_return_ok then
--     print('Failed to scale object size!!! Error Code: ', rs)
--   end
--   return rs
-- end


-- Randomize scene colors
function vrep:randSceneColor()
  local rs,outInts,outFloats,outStrings,outBuffer=simxCallScriptFunction(self.clientID, "Baxter_leftArm_joint1", sim_scripttype_childscript, "randSceneColors", {}, {}, {}, "", simx_opmode_blocking)
  if rs ~= simx_return_ok then
    print('Failed to randomize scene colors!!! Error Code: ', rs)
  elseif #outInts == torch.Tensor(outInts):sum() then
    rs = 1
    print("Scene Colors Randomized!")
  end
  return rs
end

-- Randomize clutter objects
function vrep:randClutter()
  local rs,outInts,outFloats,outStrings,outBuffer=simxCallScriptFunction(self.clientID, "Baxter_leftArm_joint1", sim_scripttype_childscript, "randClutter", {}, {}, {}, "", simx_opmode_blocking)
  if rs ~= simx_return_ok then
    print('Failed to randomize clutter objects!!! Error Code: ', rs)
  elseif #outInts == torch.Tensor(outInts):sum() then
    rs = 1
    print("Clutter Objects Randomized!")
  end
  return rs
end

-- Randomize FoV
function vrep:randFoV()
  local rs,outInts,outFloats,outStrings,outBuffer=simxCallScriptFunction(self.clientID, "Baxter_leftArm_joint1", sim_scripttype_childscript, "randFoV", {}, {}, {}, "", simx_opmode_blocking)
  if rs ~= simx_return_ok then
    print('Failed to randomize FoV!!! Error Code: ', rs)
  elseif #outInts == torch.Tensor(outInts):sum() then
    rs = 1
    print("FoV Randomized!")
    print("FoV:",outFloats[1])
  end
  return rs
end

-- Randomize FoV
function vrep:setDomainRandParameters(DR_scene_color_var,DR_camera_FoV_var,ManiRegion_min,ManiRegion_max)
  local inFloats = {DR_scene_color_var,DR_camera_FoV_var,unpack(ManiRegion_min),unpack(ManiRegion_max)}
  local rs,outInts,outFloats,outStrings,outBuffer=simxCallScriptFunction(self.clientID, "Baxter_leftArm_joint1", sim_scripttype_childscript, "setDomainRandParameters", {}, inFloats, {}, "", simx_opmode_blocking)
  if rs ~= simx_return_ok then
    print('Failed to set domain randomization parameters!!! Error Code: ', rs)
  else
    local diff = torch.Tensor(outFloats):csub(torch.Tensor(inFloats))
    if diff:sum() == 0 then
      rs = 1
      print("Parameters Setting Succeeded!")
    end
  end
  return rs
end
