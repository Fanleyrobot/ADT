--[[ File
@description:
    This class contains various conventional controllers.
@version: V0.10
@author: Fangyi Zhang   email:gzzhangfangyi@gmail.com
@acknowledgement:
    ARC Centre of Excellence for Robotic Vision (ACRV)
    Queensland Univsersity of Technology (QUT)
@history:
    V0.00   15/06/2016  developed the first version
    V0.10   19/05/2017  added the functions for 3D picking
    V0.20   25/05/2017  added the function for 3D picking in velocity mode
]]

require 'torch'

local ctrls = torch.class('conventional_controllers')
local modf_func = math.modf

function ctrls:__init(args)
    self.mode = args.mode or 'default'
end


-- a conventional controller for 2D target reaching
function ctrls:Target_Reaching_2D(desired_pos, current_pos, action_step)

    local n = #desired_pos
    local pos_delta = {}
    local action_delta = {}
    local binary_action = {}
    local action_index

    for i=1,n do
        pos_delta[i] = desired_pos[i] - current_pos[i]
        action_delta[i] = pos_delta[i] / action_step
        local action_int = modf_func(action_delta[i])
        if action_int >= 1 then
            binary_action[i] = 3
        elseif action_int <= -1 then
            binary_action[i] = 1
        else
            binary_action[i] = 2
        end
        action_index = 3*(i-1) + binary_action[i]
        if binary_action[i] ~= 2 then
            break
        end
    end

    return action_index
end


-- a conventional controller for 3D picking in position mode
function ctrls:Picking3DPos(desired_pos, current_pos, action_step)
  local n = #desired_pos
  local pos_delta = {}
  local action_delta = {}
  local binary_action = {}
  local no_action_num = 0

  for i=1,n do
    pos_delta[i] = desired_pos[i] - current_pos[i]
    action_delta[i] = pos_delta[i] / action_step
    local action_int = math.modf(action_delta[i])
    if action_int >= 1 then
        binary_action[i] = 3
    elseif action_int <= -1 then
        binary_action[i] = 1
    else
        binary_action[i] = 2
    end
    if binary_action[i] == 2 then
        no_action_num = no_action_num + 1
    end
  end

  local joint_index, action_index
  if no_action_num == n then
    local m = 3 * n
    action_index = math.floor(1 + torch.uniform() * m)
    if action_index > m then
      action_index = m
    end
  else
    for i=1,n do
      joint_index = math.floor(1 + torch.uniform() * n)
      if joint_index > n then
        joint_index = n
      end
      if binary_action[joint_index] ~= 2 then
          break
      end
    end
    action_index = 3*(joint_index-1) + binary_action[joint_index]
  end

  return action_index
end


-- a conventional controller for 3D picking in position mode
function ctrls:Picking3DPos_15(desired_pos, current_pos, action_step)
  local n = #desired_pos
  local m = 2 * n + 1
  local pos_delta = {}
  local action_delta = {}
  local binary_action = {}
  local no_action_num = 0

  for i=1,n do
    pos_delta[i] = desired_pos[i] - current_pos[i]
    action_delta[i] = pos_delta[i] / action_step
    local action_int = math.modf(action_delta[i])
    if action_int >= 1 then
        binary_action[i] = 2
    elseif action_int <= -1 then
        binary_action[i] = 1
    else
        binary_action[i] = 3
    end
    if binary_action[i] == 3 then
        no_action_num = no_action_num + 1
    end
  end

  local joint_index, action_index
  if no_action_num == n then
    action_index = math.floor(1 + torch.uniform() * m)
    if action_index > m then
      action_index = m
    end
  else
    for i=1,n do
      joint_index = math.floor(1 + torch.uniform() * n)
      if joint_index > n then
        joint_index = n
      end
      if binary_action[joint_index] ~= 3 then
          break
      end
    end
    if binary_action[joint_index] == 3 then
      action_index = m
    else
      action_index = 2*(joint_index-1) + binary_action[joint_index]
    end
  end

  return action_index
end


-- a conventional controller for 3D picking in velocity mode
function ctrls:Picking3DVel(desired_pos, current_pos)
  local n = #desired_pos
  -- local v_limit = {1.}
  -- local pos_delta = {}
  local vel = {}

  for i=1,n do
    vel[i] = desired_pos[i] - current_pos[i]
    if vel[i] > 1.0 then
      vel[i] = 1.0
    elseif vel[i] < -1.0 then
      vel[i] = -1.0
    end
  end

  return vel
end
