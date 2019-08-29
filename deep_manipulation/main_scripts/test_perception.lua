--[[ File
@description:
    This file is for testing percetion modules
    in either simulated or real-world scenarios
    in the form of testsets or on-the-fly tests.
@version: V0.20
@author: Fangyi Zhang   email:gzzhangfangyi@gmail.com
@acknowledgement:
    ARC Centre of Excellence for Robotic Vision (ACRV)
    Queensland Univsersity of Technology (QUT)
@history:
    V0.00   23/06/2016  developed the first version
    V0.01   29/06/2016  added the recovering of normalized angles to real values
    V0.10   02/07/2016  adapted the codes for the class of real_sim
    V0.11   04/07/2016  adapted the codes for the updated function of one_frame
    V0.12   06/07/2016  added a line to print normalized errors
    V0.13   08/07/2016  merged different testing cases
    V0.14   11/07/2016  fixed some bugs in the function of single_joint_test
    V0.15   13/07/2016  added the function of plotting a learning curve
    V0.16   30/08/2016  updated the function of rand_pose to use testset
    V0.17   15/05/2017  added the single sample test function for the test of vrep dataset
    V0.18   19/05/2017  fixed the bug in the single sample test function
    V0.19   03/06/2017  added one function for real perception test
    v0.20   23/07/2018  re-organized this script
]]

-- detect whether "dmn" has been constructed or is still nil
if not dmn then
    require "common/initenv"
end
-- require "manipulation_sim_class" -- for loading the lua manipulation simulator
require "common/visualization" -- for network visualization
require "common/utilities" -- some utilization functions
require "common/curve_plot" -- for ploting curves
require "python"
require "image"

local rospy = python.import("rospy")

-- get settings from the running script
local cmd = torch.CmdLine()
cmd:text()
cmd:text('Train Agent in Environment:')
cmd:text()
cmd:text('Options:')

cmd:option('-name', '', 'filename used for saving network and training history')
cmd:option('-network', '', 'reload pretrained network')
cmd:option('-net_start', '', 'network starting steps')
cmd:option('-net_end', '', 'network ending steps')
cmd:option('-net_step', '', 'network episode steps')
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

-- Parameters for controlling screen display and curve plotting
cmd:option('-display', 0, 'display flag') -- Set to 1 to show agent train
cmd:option('-display_avail', 0, 'display available flag') -- Checks to see if a display is available
cmd:option('-plot_filename', 'plot.png', 'filename used for saving average action value curve') --Set plot filename
-- End Addition

-- Parameters for generating a learning curve
cmd:option('-sim_file', 'manipulator_1dof', 'simulator filename') --Set simulator filename
cmd:option('-test_mode', 'statistic', 'testing mode') --Set simulator filename
cmd:option('-testset', '', 'testing set') --Set simulator filename
cmd:option('-learning_curve', 'false', 'whether to plot a learning curve') --Set simulator filename
-- End Addition

cmd:text()

-- Parse parameters into a lua table saved as 'opt'
local opt = cmd:parse(arg)

--- General setup.
-- override print to always flush the output
local old_print = print
local print = function(...)
    old_print(...)
    io.flush()
end

local game_actions, agent

if opt.test_mode ~= "learning_curve" then
  game_actions, agent, opt = setup(opt)
end

-- TODO: Adapt the visualization class for various cases
-- Initialize a visulization object for the network
local visu_weight = true
-- local input_dims = agent.input_dims
-- local visu_ob = visualization{network=agent.network,
--                                 input_dims=input_dims,
--                                 visu_weight=visu_weight,
--                                 options=opt}

-- Load manipulator
local manip_ob = require(opt.sim_file)


-- A function for evaluating the performance of a perception module using a test set.
function statistic_test(agent_, manip_ob_)
    local screen, label = manip_ob_:get_batch(500)
    screen = agent_:preprocess(screen):float()
    local cost, cost_sum = agent_:compute_validation_statistics(screen, label)
    print("================================================")
    print("cost: ", cost)
    print("cost_sum: ", cost_sum)
    return screen[1]
end

-- A function for testing each single joint angle detection in planar reaching
function single_joint_test(agent_, manip_ob_)
    -- Set initial joint configurations
    if not start_pose then
        start_pose = {math.pi/2, -math.pi/2, 0.0, -math.pi/2, math.pi/2, -math.pi/2, math.pi/2}
    end
    -- Set a starting direction to move one joint
    if not direc then
        direc = 1
    end
    local joint_index = 5     -- Set joint index to rotate
    local joint_step = 0.04   -- Set joint step length
    local all_n = 25          -- Set the number of ratations to test

    -- Test the single-joint cases
    if start_pose[joint_index] + all_n*joint_step >= manip_ob.joint_pose_max[joint_index] then
        direc = -1
    end
    if start_pose[joint_index] - all_n*joint_step <= manip_ob.joint_pose_min[joint_index] then
        direc = 1
    end
    start_pose[joint_index] = start_pose[joint_index] + direc * joint_step

    -- Get the screenshot and label for current arm configuration
    -- local start_target = manip_ob_:getEndEffectorPos(start_pose)
    -- local screen, label = manip_ob_:one_frame(start_pose,start_target)
    local screen, label = manip_ob_:one_frame(start_pose,nil)

    -- Estimate the arm pose for current screenshot
    local predicted_pos = agent_:test(screen) -- get the predicted low-dimensional features
    print("normalized error: ", torch.csub(predicted_pos,label))

    -- Denormalize the predicted arm pose
    predicted_pos = manip_ob_:reallignLowDimTensor(predicted_pos[1])
    predicted_pos = manip_ob_:recoverLowDimTensor(predicted_pos) -- recover to original values
    local true_pos = manip_ob_:reallignLowDimTensor(label)
    true_pos = manip_ob_:recoverLowDimTensor(true_pos)

    -- print("ground truth: ", torch.Tensor(robot_configuration.current_pos))
    -- print("predicted: ", torch.Tensor(current_pos))
    -- print("predicted: ", torch.Tensor(current_pos):sub(4,5))
    -- print("predicted: ", predicted_pos)
    -- print("ground_truth: ", true_pos)
    print("original error: ", predicted_pos:csub(true_pos))

    return screen

end

-- A function to recalibrate the pixel values to the inverval [0,1]
function image_minmax(im_)
    local im_min = im_:min()
    local im_max = im_:max()
    local normalized_im = im_:add(-im_min):div(im_max-im_min)

    return normalized_im
end

-- A function for testing a perception module using a single frame
function single_sample_test(agent_, manip_ob_)

    -- local start_target = manip_ob_:getEndEffectorPos(start_pose)
    -- local screen, label = manip_ob_:one_frame(start_pose,start_target)
    local screen, label = manip_ob_:get_one_sample()
    local true_pos = torch.Tensor(label)
    label = manip_ob_:outputLowDimTensor(nil, label)
    -- print("normalized label: ", label)

    -- screen = image_minmax(screen)
    -- print("Max:",screen:max())
    -- print("Min:",screen:min())

    -- select an action to take
    local predicted_pos = agent_:test(screen) -- get the predicted low-dimensional features
    print("===================================")
    print("normalized error: ", torch.csub(predicted_pos,label))

    predicted_pos = manip_ob_:recoverLowDimTensor(predicted_pos) -- recover to original values

    -- print("ground truth: ", torch.Tensor(robot_configuration.current_pos))
    -- print("predicted: ", torch.Tensor(current_pos))
    -- print("predicted: ", torch.Tensor(current_pos):sub(4,5))
    print("predicted: ", predicted_pos)
    -- print("ground_truth: ", true_pos)
    print("original error: ", predicted_pos:csub(true_pos))

    return screen

end

-- A function for testing a perception module using a single frame
-- but without labels, e.g., in the real world
function single_sample_test_real(agent_, manip_ob_)

    -- local start_target = manip_ob_:getEndEffectorPos(start_pose)
    -- local screen, label = manip_ob_:one_frame(start_pose,start_target)
    local screen = manip_ob_:get_one_frame()
    -- local true_pos = torch.Tensor(label)
    -- label = manip_ob_:outputLowDimTensor(nil, label)
    -- print("normalized label: ", label)

    -- select an action to take
    local predicted_pos = agent_:test(screen) -- get the predicted low-dimensional features

    predicted_pos = manip_ob_:recoverLowDimTensor(predicted_pos) -- recover to original values

    print("===================================")
    -- print("ground truth: ", torch.Tensor(robot_configuration.current_pos))
    -- print("predicted: ", torch.Tensor(current_pos))
    -- print("predicted: ", torch.Tensor(current_pos):sub(4,5))
    print("predicted: ", predicted_pos)
    -- print("original error: ", predicted_pos:csub(true_pos))

    return screen

end

-- A function for evaluating the performance of a perception module using a test set
-- the old version, which calculate in a minibatch manner
function test_set_statistic_old(agent_, manip_ob_, testset_)

    if not testset_s then
        testset_s = load_pre_constructed_dataset(testset_)
        s_screen = testset_s.image
        l_label = testset_s.label
        -- l_label = testset_s.object_position
    end

    local screen, label = manip_ob_:get_batch(400)

    screen = agent_:preprocess(screen):float()
    local p = agent_.network:forward(screen):float()

    -- Compute error = p - l
    local delta = p:clone():float()
    delta:add(-1, label)
    delta = delta:float()
    local v_mu = torch.mean(delta,1)
    local v_sigma = torch.std(delta,1)
    local e_2 = torch.pow(delta,2)
    local e = torch.sum(e_2,2)
    e:sqrt()
    local mu = torch.mean(e)
    local sigma = torch.std(e)

    print("================================================")
    print("norm: ", mu)
    print("norm_std: ", sigma)
    print("e: ", v_mu)
    print("e_std: ", v_sigma)
    -- print("norm tensor:",e)

    return screen[1]
end

-- A function for evaluating the performance of a perception module using a test set
-- the latest version, which calculate frame by frame in a for loop
function test_set_statistic(agent_, manip_ob_, testset_)

    if not testset_s then
        testset_s = load_pre_constructed_dataset(testset_)
        s_screen = testset_s.image
        -- l_label = testset_s.label
        l_label = testset_s.object_position
        if l_label == nil then
            l_label = testset_s.label
        end
        sample_amount = testset_s.sample_amount
        -- if sample_amount > 20 then
        --   sample_amount = 20
        -- end
    end

    local screen = torch.Tensor(sample_amount, unpack(s_screen[1]:size():totable()))
    local label = torch.Tensor(sample_amount, unpack(torch.Tensor(l_label[1]):size():totable()))
    local p = torch.Tensor(sample_amount, 3)
    for i=1,sample_amount do
        screen[i] = s_screen[i]
        -- screen[i] = manip_ob_:image_normalization(screen[i])
        local l_temp = table.copy(l_label[i])
        l_temp[3] = l_temp[3] + 0.0325
        label[i] = manip_ob_:outputLowDimTensor(nil,l_temp)
        local p_one_frame = agent_:test(screen[i]):float()

        if p_one_frame:size(1) > 3 then
          p_one_frame = p_one_frame:narrow(1,1,3)
        end

        p[i] = p_one_frame
        -- image.save("test"..i..".png", screen[i])
        -- Save each test images for analysis
        -- visu_ob:screen_shots(screen[i], i, true)
    end

    -- screen = agent_:preprocess(screen):float()
    -- local p = agent_.network:forward(screen):float()

    -- local p = agent_:test(screen):float()
    --
    -- if p:size(2) > 3 then
    --   p = p:narrow(2,1,3)
    -- end
    -- Compute error = p - l
    local delta = p:clone():float()
    delta:add(-1, label)
    delta = delta:float()
    local v_mu = torch.mean(delta,1)
    local v_sigma = torch.std(delta,1)
    local e_2 = torch.pow(delta,2)
    local e = torch.sum(e_2,2)
    e:sqrt()
    local mu = torch.mean(e)
    local sigma = torch.std(e)
    local med, _ = torch.median(e,1)
    med = med[1][1]

    print("-----------------------------------------------")
    print("median: ",med)
    print("norm: ", mu)
    print("norm_std: ", sigma)
    print("e: ", v_mu)
    print("e_std: ", v_sigma)
    print("norm tensor:",e)
    print("================================================")

    return screen[1], {med=med,mu=mu,sigma=sigma}
end

-- A function for evaluating perception modules in different training steps
-- to plot a learning curve
function get_learning_curve(manip_ob_, testset_, opt_)

  --- General setup.
  local network_prefix = opt_.network
  local start_agent = opt_.net_start or 5000
  local end_agent = opt_.net_end or 30000
  local sample_step = opt_.net_step or 5000
  local n_samples = 1 + (end_agent - start_agent) / sample_step
  n_samples = math.floor(n_samples)

  local med = {}
  local mu = {}
  local sigma = {}
  for i=1,n_samples do
    local network_postfix = start_agent + (i-1)*sample_step
    local folder = ""
    opt_.network=folder .. network_prefix .. network_postfix .. ".t7"
    print("Network: ",opt_.network)
    local cur_game_actions, cur_agent, cur_opt = setup(opt_)
    local _,results = test_set_statistic(cur_agent, manip_ob_, testset_)
    med[i] = results.med
    mu[i] = results.mu
    sigma[i] = results.sigma

    print(results.med)
    print(results.mu)
    print(results.sigma)

    collectgarbage()

  end

  local med_tensor = torch.Tensor(med)
  local mu_tensor = torch.Tensor(mu)
  local sigma_tensor = torch.Tensor(sigma)
  pcall(plotTensorToFile, med_tensor, 'Median history', 'Training Epochs', 'Median', network_prefix.."med_"..start_agent.."-"..end_agent..".png")
  pcall(plotTensorToFile, mu_tensor, 'Mean history', 'Training Epochs', 'Mean', network_prefix.."mu_"..start_agent.."-"..end_agent..".png")
  pcall(plotTensorToFile, sigma_tensor, 'Std history', 'Training Epochs', 'Std', network_prefix.."sigma_"..start_agent.."-"..end_agent..".png")

  print("================================================")
  print("Median: ", med_tensor)
  print("Mean: ", mu_tensor)
  print("Std: ", sigma_tensor)

end


-- A function for testing an agent for learning a Lyapunov function
function test_set_statistic_Lyapunov(agent_, manip_ob_, testset_)

    if not testset_s then
        testset_s = load_pre_constructed_dataset(testset_)
        s_screen = testset_s.image
        s_q = testset_s.arm_pose
        -- l_label = testset_s.label
        l_label = testset_s.distance_e2t
        sample_amount = testset_s.sample_amount
        -- if sample_amount > 20 then
        --   sample_amount = 20
        -- end
    end

    local screen = torch.Tensor(sample_amount, unpack(s_screen[1]:size():totable()))
    local q = torch.Tensor(sample_amount, unpack(torch.Tensor(s_q[1]):size():totable()))
    local label = torch.Tensor(sample_amount, unpack(torch.Tensor(l_label[1]):size():totable()))
    for i=1,sample_amount do
        screen[i] = s_screen[i]:clone()
        -- screen[i] = manip_ob_:image_normalization(screen[i])
        local q_temp = table.copy(s_q[i])
        q[i] = manip_ob_:outputLowDimTensor(q_temp,nil)
        label[i] = torch.Tensor({l_label[i]})
        -- Save each test images for analysis
        -- visu_ob:screen_shots(screen[i], i, true)
    end

    s = agent_:float(agent_:preprocess({q,screen}))
    local p = agent_:float(agent_.network:forward(s))

    -- Compute error = p - l
    local delta = p:clone():float()
    delta:add(-1, label)
    delta = delta:float()
    local v_mu = torch.mean(delta,1)
    local v_sigma = torch.std(delta,1)
    local e_2 = torch.pow(delta,2)
    local e = torch.sum(e_2,2)
    e:sqrt()
    local mu = torch.mean(e)
    local sigma = torch.std(e)

    print("================================================")
    print("norm: ", mu)
    print("norm_std: ", sigma)
    print("e: ", v_mu)
    print("e_std: ", v_sigma)
    print("norm tensor:",e)

    return screen[1]
end



-- A function for testing a perception module with random poses in a pose set
local testset
local test_pose
local test_target
local nepisodes
-- -- local testindex = {45,122,124,129,187,188,190,192,211,239,243,245,246,248,249,251,274,279,291,297,298,299,307,308,312,313,314,317,321,323,326,350,353}
local testindex = {45,122,187,243,245,248,274,279,297,298,299,307,308,312,313,314,317,321,323,326,350,353,113,193,200,208,202,208,233,236,239,251,280,283,293,300,302,328,332,343}
-- -- local testindex = {297}
function rand_pose_test(agent_, manip_ob_)
    if not testset then
        testset = load_pre_constructed_dataset("3DoF_testset_pose_target.t7")
        test_pose = testset.pose
        test_target = exp.target
        nepisodes = 0
    end

    local screen, label
    if manip_ob_.dataset then
        screen, label = manip_ob_:get_batch(1)
        screen = screen[1]
        label = label[1]
    else
        print("Sample:",testindex[nepisodes+1])
        screen, label = manip_ob_:one_frame(test_pose[testindex[nepisodes+1]], test_target[testindex[nepisodes+1]])
        nepisodes = nepisodes + 1
    end

    -- select an action to take
    local predicted_pos = agent_:test(screen) -- get the predicted low-dimensional features
    print("===============================")
    print("prediction:", predicted_pos)
    print("normalized label:", label)
    print("normalized error: ", torch.csub(predicted_pos,label))
    print("------------------------------")

    predicted_pos = manip_ob_:reallignLowDimTensor(predicted_pos[1])
    predicted_pos = manip_ob_:recoverLowDimTensor(predicted_pos) -- recover to original values
    local true_pos = manip_ob_:reallignLowDimTensor(label)
    true_pos = manip_ob_:recoverLowDimTensor(true_pos)

    -- print("ground truth: ", torch.Tensor(robot_configuration.current_pos))
    -- print("predicted: ", torch.Tensor(current_pos))
    -- print("predicted: ", torch.Tensor(current_pos):sub(4,5))
    -- print("predicted: ", predicted_pos)
    -- print("ground_truth: ", true_pos)
    print("original prediction:", predicted_pos)
    print("true label:", true_pos)
    print("original error: ", predicted_pos:csub(true_pos))

    return screen
end


-- A function for ploting a learning curve using the data stored in an agent history
function plot_learning_curve(cost_history)
    local cost_sum_history_tensor = torch.Tensor(cost_history)
    if pcall(plotTensorToFile, cost_sum_history_tensor, 'cost sum history', 'Training Epochs', 'cost sum', opt.plot_filename..".png") then
        -- no errors
    else
        print("plotTensor raised an error")
    end
end


-- MAIN CODES --

-- Plot a learning curve for the history cost_sum saved in a pre-trained agent
if opt.learning_curve == "true" and opt.test_mode ~= "learning_curve" then
    print("plotting:", opt.learning_curve)
    -- require "common/curve_plot"
    plot_learning_curve(agent.cost_sum_history)
end


local test_mode = opt.test_mode
local step = 0
-- Initialize performance assessment
while step < opt.steps and not rospy.is_shutdown() do
    step = step + 1

    local screen
    if test_mode == "statistic" then
        screen = statistic_test(agent, manip_ob)
    elseif test_mode == "single_joint" then
        screen = single_joint_test(agent, manip_ob)
    elseif test_mode == "single_sample" then
        screen = single_sample_test(agent, manip_ob)
    elseif test_mode == "single_sample_real" then
        screen = single_sample_test_real(agent, manip_ob)
    elseif test_mode == "rand_pose" then
        screen = rand_pose_test(agent, manip_ob)
    elseif test_mode == "testset" then
      -- screen = test_set_statistic_old(agent, manip_ob, opt.testset)
        screen, _ = test_set_statistic(agent, manip_ob, opt.testset)
    elseif test_mode == "learning_curve" then
        get_learning_curve(manip_ob, opt.testset, opt)
    end

    -- visualize but does not save files
    -- visu_ob:screen_shots(screen, step, true)
    -- visu_ob:nn_outputs_and_weights(step, true)

    -- collect memory garbages to keep it effective regarding the memory usage
    if step%10000 == 0 then collectgarbage() end

    if step == 4000 or test_mode == "testset" or test_mode == "learning_curve" then
        break
    end

end

print("Testing Done!")
print("No. of tested samples: ", step)
