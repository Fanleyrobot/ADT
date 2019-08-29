--[[ File
@description:
    This file is for training a perception module using supervised learning with simulated data.
@version: V0.22
@author: Fangyi Zhang   email:gzzhangfangyi@gmail.com
@acknowledgement:
    ARC Centre of Excellence for Robotic Vision (ACRV)
    Queensland Univsersity of Technology (QUT)
@history:
    V0.00   03/06/2016  developed the first version
    V0.10   06/06/2016  changed to save all network with FloatTensor
    V0.20   09/06/2016  fixed the bug of failing to plot curves
    V0.21   09/06/2016  reduced the network file by clearing intermedia states before saving
    V0.22   23/06/2016  added the case for real-world images
]]

-- detect whether "dmn" has been constructed or is still nil
if not dmn then
    require "common/initenv"
end
-- require "manipulation_sim_class" -- Added by Fangyi Zhang, for loading the lua manipulation simulator
require "common/visualization" -- Added by Fangyi Zhang, for network visualization
require "common/curve_plot" -- Added by Fangyi Zhang, for ploting curves

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
cmd:option('-real_dataset', 'nil', 'load a real-world dataset')
cmd:option('-real_ratio', 0.5, 'the ratio of real images in a minibatch')
-- End Addition

cmd:text()

local opt = cmd:parse(arg)

--- General setup.
local game_actions, agent, opt = setup(opt)

-- Initialize a visulization object for the network
local visu_weight = true
local input_dims = agent.input_dims
local visu_ob = visualization{network=agent.network,
                                input_dims=input_dims,
                                visu_weight=visu_weight,
                                options=opt}

-- Initialize a simulation object
local sim_ob = require(opt.sim_file)
-- manipulation_sim{step_interval=1}

-- Initialize a real_sim object
local real_ob, real_ratio
if opt.real_dataset~='nil' then
    require 'real_sim'
    local args = {}
    args.joint_end_index = {3, 5, 7} -- the joint index of the controlable joints
    args.dataset = opt.real_dataset
    real_ob = real_sim(args)
    real_ratio = opt.real_ratio
end

-- override print to always flush the output
local old_print = print
local print = function(...)
    old_print(...)
    io.flush()
end

-- definitions for the game process control and statisctic
local step = 0 -- step counting variable for the main while loop
local valiset_size = agent.n_vali_set -- the size of the validation set
local minibatch_size = agent.minibatch_size -- minibatch size

-- Sample validation set
local s, l
if real_ob then
    local n_r = math.floor(valiset_size*real_ratio)
    -- print("real amounts: ", n_r)
    local s0_r, s_r, l_r = real_ob:get_batch(n_r)
    local s_s, l_s = sim_ob:get_batch(valiset_size-n_r)
    s = torch.cat(s_r,s_s,1)
    l = torch.cat(l_r,l_s,1)
else
    s, l = sim_ob:get_batch(valiset_size)
end
-- require 'image'
-- print("total: ", valiset_size)
-- for i=1,valiset_size do
--     visu_ob:screen_shots(s[i], i, true)
-- end
agent:sample_validation_data(s, l)
print("validation sampling finished")
print("Validation set size: (s, l)", #s, #l)
print("s (min max):", s[1]:min(), s[1]:max())
print("l (min max):", l[1]:min(), l[1]:max())

while step < opt.steps do
    step = step + 1

    -- Training
    local s, l
    if real_ob then
        local n_r = math.floor(minibatch_size*real_ratio)
        local s0_r, s_r, l_r = real_ob:get_batch(n_r)
        local s_s, l_s = sim_ob:get_batch(minibatch_size-n_r)
        s = torch.cat(s_r,s_s,1)
        l = torch.cat(l_r,l_s,1)
    else
        s, l = sim_ob:get_batch(minibatch_size)
    end
    agent:train(s, l)

    -- Output weight and gradient periodically
    if step % opt.prog_freq == 0 then -- opt.prog_freq is the string output frequency
        assert(step==agent.numSteps, 'trainer step: ' .. step ..
                ' & agent.numSteps: ' .. agent.numSteps)
        print("Steps: ", step)
        agent:report() -- print abs_means and abs_maxs of the network weights and gradients
        collectgarbage() -- collect memory garbages to keep it effective regarding the memory usage
    end

    -- collect memory garbages to keep it effective regarding the memory usage
    if step%1000 == 0 then collectgarbage() end

    -- Save images periodically
    if step % opt.eval_freq > 0 and  step % opt.eval_freq <= 1000 then
        -- visualize but does not save files
        visu_ob:nn_outputs_and_weights(step, true)
    end

    -- Validate the results
    if step % opt.eval_freq == 0 then
        local cost, cost_sum = agent:compute_validation_statistics()
        print("cost: ", cost)
        print("cost sum: ", cost_sum)

        local cost_sum_history_tensor = torch.Tensor(agent.cost_sum_history)
        if (opt.display_avail == 1) then
            if (opt.display == 1) then
                if pcall(plotTensorToDisplay, cost_sum_history_tensor, 'cost sum history', 'Training Epochs', 'cost sum', 1) then
                    -- no errors
                else
                    -- plotTensor raised an error
                    print("plotTensor raised an error")
                end
            end
        end

        if pcall(plotTensorToFile, cost_sum_history_tensor, 'cost sum history', 'Training Epochs', 'cost sum', opt.plot_filename..".png") then
            -- no errors
        else
            -- plotTensor raised an error
            print("plotTensor raised an error")
        end
    end

    -- save the neural network and some history data
    if step % opt.save_freq == 0 or step == opt.steps then
        local filename_ = opt.name .. "_" .. step
        local model = agent.network:clone()
        model = model:clearState():float() -- clear intermedia states and convert to a CPU model
        torch.save(filename_ .. ".t7", {model = model,
                                cost_history = agent.cost_history,
                                cost_sum_history = agent.cost_sum_history})
        print('Saved:', filename_ .. '.t7')
        io.flush()
        collectgarbage()
    end

end
