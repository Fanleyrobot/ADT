--[[ File
@description:
    This file is for testing a connected end-to-end network.
@version: V0.00
@author: Fangyi Zhang   email:gzzhangfangyi@gmail.com
@acknowledgement:
    ARC Centre of Excellence for Robotic Vision (ACRV)
    Queensland Univsersity of Technology (QUT)
@history:
    V0.00   21/06/2016  developed the first version
]]

-- detect whether "dmn" has been constructed or is still nil
if not dmn then
    require "common/initenv"
end
-- require "manipulation_sim_class" -- Added by Fangyi Zhang, for loading the lua manipulation simulator
require "common/visualization" -- Added by Fangyi Zhang, for network visualization
require "common/evaluation" -- Added by Fangyi Zhang, for evaluation performance

-- get settings from the running script
local cmd = torch.CmdLine()
cmd:text()
cmd:text('Train Agent in Environment:')
cmd:text()
cmd:text('Options:')

cmd:option('-name', '', 'filename used for saving network and training history')
cmd:option('-network', '', 'reload pretrained network')
cmd:option('-agent', '', 'name of agent file to use')
cmd:option('-agent_params', '', 'string of agent parameters')
cmd:option('-seed', 1, 'fixed input seed for repeatable experiments')
cmd:option('-saveNetworkParams', false,
           'saves the agent network in a separate file')
cmd:option('-prog_freq', 5*10^3, 'frequency of progress output')
cmd:option('-save_freq', 5*10^4, 'the model is saved every save_freq steps')
cmd:option('-eval_freq', 10^4, 'frequency of greedy evaluation')
cmd:option('-save_versions', 0, '')

cmd:option('-steps', 10^5, 'number of training steps to perform')

cmd:option('-verbose', 2,
           'the higher the level, the more information is printed to screen')
cmd:option('-threads', 1, 'number of BLAS threads')
cmd:option('-gpu', -1, 'gpu flag')

-- Added parameters for controlling screen display and curve plotting
cmd:option('-display', 0, 'display flag') -- Set to 1 to show agent train
cmd:option('-display_avail', 0, 'display available flag') -- Checks to see if a display is available
cmd:option('-plot_filename', 'plot.png', 'filename used for saving average action value curve') --Set plot filename
-- End Addition

-- Added parameters
cmd:option('-sim_file', 'manipulator_1dof', 'simulator filename') --Set simulator filename
cmd:option('-testset', '', 'reload testset')
cmd:option('-feature_mode', 'high', 'determine the input feature mode: low or high') --determine the input feature mode: low or high
-- End Addition

cmd:text()

local opt = cmd:parse(arg)

--- General setup.
local game_actions, agent, opt = setup(opt)

-- agent.network:clearState()
-- torch.save("transferred.t7", {model = agent.network,
--                                 cost_history = agent.cost_history,
--                                 cost_sum_history = agent.cost_sum_history})
-- print('Saved:', "transferred.t7")

-- Initialize a visulization object for the network
local visu_weight = true
local input_dims = agent.input_dims
-- local visu_ob = visualization{network=agent.network,
--                                 input_dims=input_dims,
--                                 visu_weight=visu_weight,
--                                 options=opt}

-- Initialize a simulation object
local sim_ob = require(opt.sim_file)
-- manipulation_sim{step_interval=1}

-- Initialize an evaluation object for the testing
local evalu_ob = evaluation{agent=agent, options=opt}

-- override print to always flush the output
local old_print = print
local print = function(...)
    old_print(...)
    io.flush()
end

require "python"
rospy = python.import("rospy")

-- definitions for the game process control and statisctic
local nepisodes = 0 -- count the number of games it played in the entire evaluation process
local step = 0 -- step counting variable for the main while loop

-- initialize the manipulator
-- sim_ob.vrep:setImageRendering(true)
local screen, reward, terminal, completion, low_dim_features, configuration, closest_distance, distance = sim_ob:picking(_, true, true)
-- Added by Fangyi Zhang to print the input image format
-- The input image tensor should be 3D or 4D tensor: (1*)3*h*w
-- The pixel value should be in the 0-1 scale
print("Screen_Size:", screen:size())
-- print("Screen_min:", screen:min())
-- print("Screen_max:", screen:max())
-- End Addition

local low_dim_mode
if opt.feature_mode == "low" then
    low_dim_mode = true
else
    low_dim_mode = false
end

-- Initialize performance assessment
local sample_amount = 20
if sim_ob.test_dataset then
  sample_amount = sim_ob.test_dataset_sample_amount
end
print("Samples to test: ", sample_amount)
local steps_in_one_trial = 0
local last_step = 0
local closest_distance_hist = torch.Tensor(sample_amount):float()
local distance_hist = torch.Tensor(sample_amount):float()
local steps_hist = torch.Tensor(sample_amount):float()
evalu_ob:performance_assessment(reward, terminal, completion, "initialize")
while step < opt.steps and not rospy.is_shutdown() do
    step = step + 1

    local features
    if low_dim_mode then
        features = low_dim_features
    else
        features = screen
    end

    -- select an action to take
    local action = agent:test(features)
    -- local action = agent:test({low_dim_features, screen})
    -- print("Arm pose: ", low_dim_features)
    -- print("Action: ", action)
    -- print("Reward: ", reward)

    -- visualize but does not save files
    -- visu_ob:screen_shots(screen, step, true)
    -- visu_ob:nn_outputs_and_weights(step, true)
    win_input = image.display{image=screen, win=win_input}


    -- game over? get next game!
    if not terminal then
        screen,reward, terminal, completion, low_dim_features, configuration, closest_distance, distance = sim_ob:picking(action:totable(), false, true)
        -- Count performance
        evalu_ob:performance_assessment(reward, terminal, completion, "counting")
    else
        steps_in_one_trial = step - last_step - 1
        last_step = step
        print("Completion:", completion)
        print("Closest_distance:", closest_distance)
        print("Final_distance:", distance)
        print("Step cost:", steps_in_one_trial)
        print("-------------------------------")
        nepisodes = nepisodes + 1
        closest_distance_hist[nepisodes] = closest_distance
        distance_hist[nepisodes] = distance
        steps_hist[nepisodes] = steps_in_one_trial
        if nepisodes >= sample_amount then
            local step_mu = torch.mean(steps_hist)
            local d_mu = torch.mean(distance_hist)
            local d_sigma = torch.std(distance_hist)
            local d_median = torch.median(distance_hist)

            -- Print the number of tested episodes
            print("======================================")
            print("Tested Episodes: ", nepisodes)
            print("Tested Samples:", #closest_distance_hist)
            print("d_median:", d_median)
            print("d_mu:", d_mu)
            print("d_sigma", d_sigma)
            print("step_mu", step_mu)
            print("distance list:", torch.Tensor(distance_hist))

            -- Save the sorted data into a new t7 file
            local filename = opt.plot_filename .. '.t7'
            torch.save(filename, {closest_distance = closest_distance_hist,
                                    distance = distance_hist,
                                    steps = steps_hist})
            print("Results Saved:", filename)

            -- Calculate performance assessment results
            evalu_ob:performance_assessment(reward, terminal, completion, "end")

            break
        end
        screen, reward, terminal, completion, low_dim_features, configuration, closest_distance, distance  = sim_ob:picking(_, true, true)
        print("Sample:", nepisodes+1)
    end

    -- collect memory garbages to keep it effective regarding the memory usage
    if step%10000 == 0 then collectgarbage() end

end
