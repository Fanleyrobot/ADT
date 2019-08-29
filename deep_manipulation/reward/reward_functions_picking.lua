--[[ File
@description:
    This file contains several different reward functions for the task of target reaching.
@version: V0.25
@author: Fangyi Zhang   email:gzzhangfangyi@gmail.com
@acknowledgement:
    ARC Centre of Excellence for Robotic Vision (ACRV)
    Queensland Univsersity of Technology (QUT)
@history:
    V0.00   25/11/2015  developed the first version
    V0.10   29/11/2015  fixed bug that the resolution of the reward function is higher than that in DQN
    V0.20   18/12/2015  added a new reward function "reward_continuous_maximum_step_limit"
    V0.21   13/01/2016  added a new reward function "reward_continuous_more_assistance_termination"
    V0.22   13/01/2016  added a reward function for testing evaluation
    V0.23   30/08/2016  updated the reword function for testing to return closest distance and final distance.
    V0.24   31/05/2017  added reward functions for 7DoF tabletop object picking
    V0.25   31/05/2017  added one parameter to set extra workspace limit (workspace_limit)
]]


require 'torch'


-- construct a class
local rwd = torch.class('reward_functions_picking')


--[[ Function
@description: initialize an object for reward functions
@input:
    args: settings for a reward function object
@output: nil
@notes:
]]
function rwd:__init(args)

    -- reward function variables
    self.max_step = args.max_step or 300 -- the maximum step limitation to complete a task
    self.step_interval = args.step_interval or 4 -- the assessment steps, if it is 10, it means the distance based assessment will be conducted every 10 steps
    self.assistance_interval = args.assistance_interval or 80 -- the assistance guidance interval
    self.completion_restart_threshold = args.completion_restart_threshold or 1 -- the threshold steps for completion restart
    self.completion_threshold = args.completion_threshold or 1 -- the threshold steps for completion
    self.initial_distance = 1 -- the distance between the end-effector and the destination at the beginning of each new game
    self.closest_distance = 100 -- the closest distance
    self.history_dis = {} -- history distance table
    self.history_gradient = {} -- history distance gradient table
    self.step_count = 0
    self.completion_count = 0
    self.completion_restart = 0
    self.completion = false -- completion flag of the current trial
    self.default_reward = -0.0005 -- the reward value when no special reward value is activated
    self.target_reaching_allowance = 0.005 -- the radius for the zone when the end-effector reaches, it will be treated as reaching the target
    self.completion_allowance = 0.04 -- the radius for the zone when the end-effector reaches, it will be treated as having completed the target reaching task
    -- self.workspace_limit = args.workspace_limit or 0.4
    self.workspace_limit = args.workspace_limit or 1.0
    print("step interval: ", self.step_interval)
    print("assistance_interval: ", self.assistance_interval)
    print("max step: ", self.max_step)

end

--[[ Function
@description: decide the sign of a number
@input:
    x: the number, i.e., -2
@output:
    sign: 1: x>0; -1: x<0; 0:others
@notes:
]]
function rwd:sign(x)
  return x>0 and 1 or x<0 and -1 or 0
end


--[[ Function
@description: compute the reward of an action
@input:
    destination: the 2D coordinate of the destination in physical models, i.e., {3.0, 3.0}
    end_effector: the 2D coordinate of the end effector in physical models, i.e., {4.0, 0.0}
    limit_reached: whether the joint limitation has been reached
@output:
    reward_: the reward value
    terminal_: whether the game is terminal, true: terminal
    completion_: whether the game has been completed, true: completed
@notes:
]]
function rwd:reward_discrete(destination, end_effector, limit_reached)
    local reward_ = 0
    local terminal_ = false
    local completion_ = false
    local distance

    -- if the joint limitation is reached, terminate the current round and return -1 reward directly
    if limit_reached then
        terminal_ = true
        reward_ = -1

    else
        distance = math.sqrt(math.pow((destination[1] - end_effector[1]), 2) + math.pow((destination[2] - end_effector[2]), 2))

        -- calculate the reward value according to current distance
        if distance > self.target_reaching_allowance then
            reward_ = self.default_reward
            self.completion_restart = 0
        else
            reward_ = 0
            self.completion_restart = self.completion_restart + 1
            if self.completion_restart > self.completion_restart_threshold then
                self.completion_restart = 0
                reward_ = 1
                terminal_ = true -- completion restart
            end
        end

        -- determine completion
        if distance <= self.completion_allowance then
            self.completion_count = self.completion_count + 1
            if self.completion_count > self.completion_threshold then
                self.completion_count = 0
                self.completion = true -- completion
                -- terminal_ = true -- completion restart
            end
        else
            self.completion_count = 0
        end

    end

    -- Reset some variables when terminating
    if terminal_ then
        completion_ = self.completion
        self.completion = false
        self.completion_restart = 0
        self.completion_count = 0
    end
    -- print("reward:", reward_)

    return reward_, terminal_, completion_
end


--[[ Function
@description: compute the reward of an action
@input:
    destination: the 2D coordinate of the destination in physical models, i.e., {3.0, 3.0}
    end_effector: the 2D coordinate of the end effector in physical models, i.e., {4.0, 0.0}
    limit_reached: whether the joint limitation has been reached
@output:
    reward_: the reward value
    terminal_: whether the game is terminal, true: terminal
    completion_: whether the game has been completed, true: completed
@notes:
]]
function rwd:reward_discrete_more_termination(destination, end_effector, limit_reached)
    local reward_ = 0
    local terminal_ = false
    local completion_ = false
    local distance

    -- if the joint limitation is reached, terminate the current round and return -1 reward directly
    if limit_reached then
        terminal_ = true
        reward_ = -1

    else
        distance = math.sqrt(math.pow((destination[1] - end_effector[1]), 2) + math.pow((destination[2] - end_effector[2]), 2))

        local m = #self.history_dis + 1
        self.history_dis[m] = distance

        -- calculate the reward value according to current distance
        if distance > self.target_reaching_allowance then
            reward_ = self.default_reward
            self.completion_restart = 0

            -- assitance guidance
            if m > self.assistance_interval then
                -- set the assistant termination condition to >= 0.4, ensuring there are some distinguishable features in images
                if self.history_dis[m] - self.history_dis[m-self.assistance_interval] >= 0 then
                    terminal_ = true
                    reward_ = -1
                end
            end

            -- self.step_count = self.step_count + 1
            -- if self.step_count >= self.assistance_interval then
            --     self.step_count = 0
            --     if self.pre_distance then
            --         if (distance - self.pre_distance) >= 0 then
            --             terminal_ = true
            --             reward_ = -1
            --         else
            --             self.pre_distance = distance
            --         end
            --     else
            --         self.pre_distance = distance
            --     end
            -- end

        else
            reward_ = 0
            self.completion_restart = self.completion_restart + 1
            if self.completion_restart > self.completion_restart_threshold then
                self.completion_restart = 0
                reward_ = 1
                terminal_ = true -- completion restart
            end

            -- when get into the reaching zone, all the assistance guidance will be reset
            -- self.step_count = 0
            -- self.pre_distance = nil

        end

        -- determine completion
        if distance <= self.completion_allowance then
            self.completion_count = self.completion_count + 1
            if self.completion_count > self.completion_threshold then
                self.completion_count = 0
                self.completion = true -- completion
                -- terminal_ = true -- completion restart
            end
        else
            self.completion_count = 0
        end

    end

    -- Reset some variables when terminating
    if terminal_ then
        completion_ = self.completion
        self.completion = false
        self.completion_restart = 0
        self.completion_count = 0
        -- self.step_count = 0
        -- self.pre_distance = nil
        self.history_dis = {}
    end
    -- print("reward:", reward_)

    return reward_, terminal_, completion_
end


--[[ Function
@description: compute the reward of an action
@input:
    destination: the 2D coordinate of the destination in physical models, i.e., {3.0, 3.0}
    end_effector: the 2D coordinate of the end effector in physical models, i.e., {4.0, 0.0}
    limit_reached: whether the joint limitation has been reached
@output:
    reward_: the reward value
    terminal_: whether the game is terminal, true: terminal
    completion_: whether the game has been completed, true: completed
@notes:
]]
function rwd:reward_continuous(destination, end_effector, limit_reached)
    local reward_ = 0
    local terminal_ = false
    local completion_ = false

    -- if the joint limitation is reached, terminate the current round and return -1 reward directly
    if limit_reached then
        terminal_ = true
        reward_ = -1

    else
        local distance = math.sqrt(math.pow((destination[1] - end_effector[1]), 2) + math.pow((destination[2] - end_effector[2]), 2))

        -- calculate the reward value according to current distance
        if distance > self.target_reaching_allowance then
            -- reduce the resolution using the floor function, due to the resolution limitation of the input images in the DQN
            -- reward_ = (self.target_reaching_allowance / math.floor(distance+1-self.target_reaching_allowance) - 1) / 1000
            reward_ = (self.target_reaching_allowance / distance - 1) / 1000
            self.completion_restart = 0
        else
            reward_ = 0
            self.completion_restart = self.completion_restart + 1
            if self.completion_restart > self.completion_restart_threshold then
                self.completion_restart = 0
                reward_ = 1
                terminal_ = true -- completion restart
            end
        end

        -- determine completion
        if distance <= self.completion_allowance then
            self.completion_count = self.completion_count + 1
            if self.completion_count > self.completion_threshold then
                self.completion_count = 0
                self.completion = true -- completion
                -- terminal_ = true -- completion restart
            end
        else
            self.completion_count = 0
        end

    end

    -- Reset some variables when terminating
    if terminal_ then
        completion_ = self.completion
        self.completion = false
        self.completion_restart = 0
        self.completion_count = 0
    end
    -- print("reward:", reward_)

    return reward_, terminal_, completion_
end


--[[ Function
@description: compute the reward of an action
@input:
    destination: the 2D coordinate of the destination in physical models, i.e., {3.0, 3.0}
    end_effector: the 2D coordinate of the end effector in physical models, i.e., {4.0, 0.0}
    limit_reached: whether the joint limitation has been reached
@output:
    reward_: the reward value
    terminal_: whether the game is terminal, true: terminal
    completion_: whether the game has been completed, true: completed
@notes:
]]
function rwd:reward_continuous_more_termination(destination, end_effector, limit_reached)
    local reward_ = 0
    local terminal_ = false
    local completion_ = false
    local distance

    -- if the joint limitation is reached, terminate the current round and return -1 reward directly
    if limit_reached then
        terminal_ = true
        reward_ = -1

    else
        distance = math.sqrt(math.pow((destination[1] - end_effector[1]), 2) + math.pow((destination[2] - end_effector[2]), 2))

        local m = #self.history_dis + 1
        self.history_dis[m] = distance

        -- calculate the reward value according to current distance
        if distance > self.target_reaching_allowance then
            -- reduce the resolution using the floor function, due to the resolution limitation of the input images in the DQN
            -- reward_ = (self.target_reaching_allowance / math.floor(distance+1-self.target_reaching_allowance) - 1) / 1000
            reward_ = (self.target_reaching_allowance / distance - 1) / 1000
            self.completion_restart = 0

            -- assitance guidance
            if m > self.assistance_interval then
                -- set the assistant termination condition to >= 0.4, ensuring there are some distinguishable features in images
                if self.history_dis[m] - self.history_dis[m-self.assistance_interval] >= 0 then
                    terminal_ = true
                    reward_ = -1
                end
            end

        else
            reward_ = 0
            self.completion_restart = self.completion_restart + 1
            if self.completion_restart > self.completion_restart_threshold then
                self.completion_restart = 0
                reward_ = 1
                terminal_ = true -- completion restart
            end

        end

        -- determine completion
        if distance <= self.completion_allowance then
            self.completion_count = self.completion_count + 1
            if self.completion_count > self.completion_threshold then
                self.completion_count = 0
                self.completion = true -- completion
                -- terminal_ = true -- completion restart
            end
        else
            self.completion_count = 0
        end

    end

    -- Reset some variables when terminating
    if terminal_ then
        completion_ = self.completion
        self.completion = false
        self.completion_restart = 0
        self.completion_count = 0
        self.history_dis = {}
    end
    -- print("reward:", reward_)

    return reward_, terminal_, completion_
end


--[[ Function
@description: compute the reward of an action
@input:
    destination: the 2D coordinate of the destination in physical models, i.e., {3.0, 3.0}
    end_effector: the 2D coordinate of the end effector in physical models, i.e., {4.0, 0.0}
    limit_reached: whether the joint limitation has been reached
@output:
    reward_: the reward value
    terminal_: whether the game is terminal, true: terminal
    completion_: whether the game has been completed, true: completed
@notes:
]]
function rwd:reward1(distance, collision, reached)
    local reward_ = 0
    local completion_ = false
    local terminal_ = false

    -- if the joint limitation is reached, terminate the current round and return -1 reward directly
    if collision or distance > self.workspace_limit then
        terminal_ = true
        reward_ = -1

    else
        local m = #self.history_dis + 1
        self.history_dis[m] = distance

        -- calculate the reward value according to current distance
        if not reached then
            -- reduce the resolution using the floor function, due to the resolution limitation of the input images in the DQN
            reward_ = (self.target_reaching_allowance / distance - 1) / 1000
            if reward_ > 0 then
              reward_ = 0
            end
            self.completion_restart = 0
            -- assitance guidance
            if m > self.assistance_interval then
                -- set the assistant termination condition to >= 0.4, ensuring there are some distinguishable features in images
                if self.history_dis[m] - self.history_dis[m-self.assistance_interval] >= 0 then
                    terminal_ = true
                    reward_ = 0.5 -- set reward_ to 0.5 as a sign to recognize the assistance termination
                end
            end

        else
            reward_ = 0
            self.completion_restart = self.completion_restart + 1
            if self.completion_restart > self.completion_restart_threshold then
                self.completion_restart = 0
                reward_ = 1
                self.completion = true
                terminal_ = true -- completion restart
            end

        end

        -- determine completion
        if distance <= self.completion_allowance then
            self.completion_count = self.completion_count + 1
            if self.completion_count > self.completion_threshold then
                self.completion_count = 0
                self.completion = true -- completion
                -- terminal_ = true -- completion restart
            end
        else
            self.completion_count = 0
        end

    end

    -- Reset some variables when terminating
    if terminal_ then
        completion_ = self.completion
        self.completion = false
        self.completion_restart = 0
        self.completion_count = 0
        self.history_dis = {}
    end
    -- print("reward:", reward_)

    return reward_, terminal_, completion_
end


--[[ Function
@description: compute the reward of an action
@input:
    destination: the 2D coordinate of the destination in physical models, i.e., {3.0, 3.0}
    end_effector: the 2D coordinate of the end effector in physical models, i.e., {4.0, 0.0}
    limit_reached: whether the joint limitation has been reached
@output:
    reward_: the reward value
    terminal_: whether the game is terminal, true: terminal
    completion_: whether the game has been completed, true: completed
@notes:
]]
function rwd:reward1_testing(distance, collision, reached)
    local reward_ = 0
    local terminal_ = false
    local completion_ = false
    local closest_distance = 100

    -- if the joint limitation is reached, terminate the current round and return -1 reward directly
    if collision or distance > self.workspace_limit then
        terminal_ = true
        reward_ = -1

    else

        if distance < self.closest_distance then
            self.closest_distance = distance
        end

        local m = #self.history_dis + 1
        self.history_dis[m] = distance

        -- calculate the reward value according to current distance
        if not reached then
            -- reduce the resolution using the floor function, due to the resolution limitation of the input images in the DQN
            -- reward_ = (self.target_reaching_allowance / math.floor(distance+1-self.target_reaching_allowance) - 1) / 1000
            reward_ = (self.target_reaching_allowance / distance - 1) / 1000
            if reward_ > 0 then
              reward_ = 0
            end
            self.completion_restart = 0
            -- maximum step limit
            if m > self.max_step then
                terminal_ = true
                -- reward_ = self.target_reaching_allowance / distance - 1
            end

        else
            reward_ = 0
            self.completion_restart = self.completion_restart + 1
            if self.completion_restart > self.completion_restart_threshold then
                self.completion_restart = 0
                reward_ = 1
                self.completion = true
                terminal_ = true -- completion restart
            end

        end

        -- determine completion
        if distance <= self.completion_allowance then
            self.completion_count = self.completion_count + 1
            if self.completion_count > self.completion_threshold then
                self.completion_count = 0
                self.completion = true -- completion
                -- terminal_ = true -- completion restart
            end
        else
            self.completion_count = 0
        end

    end

    -- Reset some variables when terminating
    if terminal_ then
        completion_ = self.completion
        self.completion = false
        self.completion_restart = 0
        self.completion_count = 0
        self.history_dis = {}
        closest_distance = self.closest_distance
        self.closest_distance = 100
    end
    -- print("reward:", reward_)

    return reward_, terminal_, completion_, closest_distance, distance
end


--[[ Function
@description: compute the reward of an action
@input:
    destination: the 2D coordinate of the destination in physical models, i.e., {3.0, 3.0}
    end_effector: the 2D coordinate of the end effector in physical models, i.e., {4.0, 0.0}
    limit_reached: whether the joint limitation has been reached
@output:
    reward_: the reward value
    terminal_: whether the game is terminal, true: terminal
    completion_: whether the game has been completed, true: completed
@notes:
]]
function rwd:reward_continuous_maximum_step_limit(destination, end_effector, limit_reached)
    local reward_ = 0
    local terminal_ = false
    local completion_ = false
    local distance

    -- if the joint limitation is reached, terminate the current round and return -1 reward directly
    if limit_reached then
        terminal_ = true
        reward_ = -1

    else
        distance = math.sqrt(math.pow((destination[1] - end_effector[1]), 2) + math.pow((destination[2] - end_effector[2]), 2))

        local m = #self.history_dis + 1
        self.history_dis[m] = distance

        -- calculate the reward value according to current distance
        if distance > self.target_reaching_allowance then
            -- reduce the resolution using the floor function, due to the resolution limitation of the input images in the DQN
            -- reward_ = (self.target_reaching_allowance / math.floor(distance+1-self.target_reaching_allowance) - 1) / 1000
            reward_ = (self.target_reaching_allowance / distance - 1) / 1000
            self.completion_restart = 0

            -- maximum step limit
            if m > self.max_step then
                terminal_ = true
                reward_ = self.target_reaching_allowance / distance - 1
            end

        else
            reward_ = 0
            self.completion_restart = self.completion_restart + 1
            if self.completion_restart > self.completion_restart_threshold then
                self.completion_restart = 0
                reward_ = 1
                terminal_ = true -- completion restart
            end

        end

        -- determine completion
        if distance <= self.completion_allowance then
            self.completion_count = self.completion_count + 1
            if self.completion_count > self.completion_threshold then
                self.completion_count = 0
                self.completion = true -- completion
                -- terminal_ = true -- completion restart
            end
        else
            self.completion_count = 0
        end

    end

    -- Reset some variables when terminating
    if terminal_ then
        completion_ = self.completion
        self.completion = false
        self.completion_restart = 0
        self.completion_count = 0
        self.history_dis = {}
    end
    -- print("reward:", reward_)

    return reward_, terminal_, completion_
end



-- =========================================================================
-- previous reward functions

--[[ Function
@description: compute the reward of an action
@input:
    destination: the 2D coordinate of the destination in physical models, i.e., {3.0, 3.0}
    end_effector: the 2D coordinate of the end effector in physical models, i.e., {4.0, 0.0}
    new: whether the game is a new game
@output:
    y: the reward value, 1: get closer; -1: get further; 0: distance unchanged
    terminal_: whether the game is terminal, true: terminal.
@notes:
]]
function rwd:reward(destination, end_effector, new)
    if new == true then -- when starting a new game, clear all history data
        self.history_dis = {}
        self.history_gradient = {}
    end

    local distance = math.sqrt(math.pow((destination[1] - end_effector[1]), 2) + math.pow((destination[2] - end_effector[2]), 2))
    --print("distance:",distance)
    self.history_dis[#self.history_dis + 1] = distance
    local m = #self.history_dis
    --print("m:",m)
    if m > 1 then
        local gradient = self.history_dis[m] - self.history_dis[m-1]
        self.history_gradient[#self.history_gradient + 1] = gradient
    end
    local n = #self.history_gradient
    local y = 0
    local terminal_ = false
    if n > 0 then
        if self.history_gradient[n] > 0 then
            y = -1
        elseif self.history_gradient[n] < 0 then
            y = 1
        end
        if n > 2 then
            local acc = 0
            for i=n-2,n do
                acc = acc + self:sign(self.history_gradient[i])
                --print("acc:",acc)
            end
            if  acc > 1 then
                terminal_ = true
            end
        end
    end
    return y, terminal_
end
