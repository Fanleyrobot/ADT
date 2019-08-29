--[[ File
@description:
    This file is for training a perception module.
@version: V0.25
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
    V0.25   05/06/2019  cleaned a bit the code
]]

-- detect whether "dmn" has been constructed or is still nil
if not dmn then
    require "common/initenv"
end
require "common/visualization" -- for network visualization
require "common/curve_plot" -- for ploting curves

-- get settings from the running script
local cmd = torch.CmdLine()
cmd:text()
cmd:text('Train Agent in Environment:')
cmd:text()
cmd:text('Options:')

cmd:option('-name', '', 'filename used for saving network and training history')
cmd:option('-sim_file', 'manipulator_1dof', 'simulator filename') --Set simulator filename, i.e., dataset file settings
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
-- Parameters for controlling screen display, curve plotting and record saving
cmd:option('-display', 0, 'display flag') -- Set to 1 to show agent train
cmd:option('-display_avail', 0, 'display available flag') -- Checks to see if a display is available
cmd:option('-record_folder', 'training_records', 'the name of a folder to store training records') --Set record folder name
cmd:option('-plot_filename', 'plot', 'filename used for saving average action value curve') --Set plot filename

cmd:text()

local opt = cmd:parse(arg)

-- General setup.
local game_actions, agent, opt = setup(opt)

-- Set and create a subfolder under the record folder to store training records
local record_folder = opt.record_folder
os.execute("mkdir " .. record_folder)
record_folder = record_folder .. '/' .. opt.plot_filename .. '/'
os.execute("mkdir " .. record_folder)

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
local s_sim, s_real_t, s_real, pose_sim, pose_real_t, pose_real = sim_ob:get_batch_dc_semi(valiset_size)
agent:sample_validation_data(s_sim, s_real, pose_sim, pose_real)
-- require 'image'
-- print("total: ", valiset_size)
-- for i=1,valiset_size do
--     visu_ob:screen_shots(s[i], i, true)
-- end

print("validation sampling finished")
print("Validation set size: (s_sim, s_real, pose_sim, pose_real)", #s_sim, #s_real, #pose_sim, #pose_real)
print("s_sim (min max):", s_sim:min(), s_sim:max())
print("s_real (min max):", s_real:min(), s_real:max())
print("pose_sim (min max):", pose_sim:min(), pose_sim:max())
print("pose_real (min max):", pose_real:min(), pose_real:max())

while step < opt.steps do
    step = step + 1

    -- Training
    s_sim, s_real_t, s_real, pose_sim, pose_real_t, pose_real = sim_ob:get_batch_dc_semi(minibatch_size)
    agent:train(s_sim, s_real_t, s_real, pose_sim, pose_real_t)
    -- agent:train(s, l, p)

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

    -- Validate the results
    if step % opt.eval_freq == 0 then
        local cost, cost_sum, d_loss, cost_sum_sim = agent:compute_validation_statistics()
        print("perception cost: ", cost)
        print("perception cost sum: ", cost_sum)
        print("sim perception cost sum: ", cost_sum_sim)
        print("discriminater loss: ", d_loss)

        local cost_sum_history_tensor = torch.Tensor(agent.hist_Lp_sum_real)
        local cost_sum_sim_history_tensor = torch.Tensor(agent.hist_Lp_sum_sim)
        local d_cost_history_tensor = torch.Tensor(agent.hist_LD)
        if (opt.display_avail == 1) then
            if (opt.display == 1) then
                pcall(plotTensorToDisplay, cost_sum_history_tensor, 'cost sum history', 'Training Epochs', 'cost sum', 1)
            end
        end
        pcall(plotTensorToFile, cost_sum_history_tensor, 'cost sum history', 'Training Epochs', 'cost sum', record_folder..opt.plot_filename..".png")
        pcall(plotTensorToFile, cost_sum_sim_history_tensor, 'cost sum history', 'Training Epochs', 'cost sum', record_folder..opt.plot_filename.."_sim.png")
        pcall(plotTensorToFile, d_cost_history_tensor, 'cost history', 'Training Epochs', 'cost', record_folder..opt.plot_filename.."_d.png")
    end

    -- save the neural network and some history data
    if step % opt.save_freq == 0 or step == opt.steps then
        local filename_ = record_folder .. opt.name .. "_" .. step
        local encoder_tg = agent.encoder_tg.network:clearState():clone()
        encoder_tg = encoder_tg:clearState():float() -- clear intermedia states and convert to a CPU model
        local encoder_sr = agent.encoder_sr.network:clearState():clone()
        encoder_sr = encoder_sr:clearState():float() -- clear intermedia states and convert to a CPU model
        local pose_net = agent.pose_net.network:clearState():clone()
        pose_net = pose_net:clearState():float() -- clear intermedia states and convert to a CPU model
        local d_net = agent.discriminater.network:clearState():clone()
        d_net = d_net:clearState():float() -- clear intermedia states and convert to a CPU model
        torch.save(filename_ .. ".t7", {encoder_tg = encoder_tg,
                                encoder_sr = encoder_sr,
                                d_net = d_net,
                                pose_net = pose_net,
                                cost_history = agent.hist_Lp_vector_real,
                                cost_sum_history = agent.hist_Lp_sum_real,
                                cost_sum_sim_history = agent.hist_Lp_sum_sim,
                                d_cost_history = agent.hist_LD})
        print('Saved:', filename_ .. '.t7')
        io.flush()
        collectgarbage()
    end

end
