--[[ File
@description:
    This file is for testing the feasibility of vrep interfaces.
@version: V0.10
@author: Fangyi Zhang   email:gzzhangfangyi@gmail.com
@acknowledgement:
    ARC Centre of Excellence for Robotic Vision (ACRV)
    Queensland Univsersity of Technology (QUT)
@history:
    V0.00   30/08/2016  developed the first version
    V0.10   24/09/2018  re-organized
]]

-- detect whether "dmn" has been constructed or is still nil
if not dmn then
    require "common/initenv"
end
-- require "manipulation_sim_class" -- Added by Fangyi Zhang, for loading the lua manipulation simulator
-- require "visualization" -- Added by Fangyi Zhang, for network visualization

-- get settings from the running script
local cmd = torch.CmdLine()
cmd:text()
cmd:text('Train Agent in Environment:')
cmd:text()
cmd:text('Options:')


-- Added parameters for controlling screen display and curve plotting
cmd:option('-display', 0, 'display flag') -- Set to 1 to show agent train
cmd:option('-sim_file', 'manipulator_1dof', 'simulator filename') --Set simulator filename
-- End Addition

cmd:text()

local opt = cmd:parse(arg)

-- Initialize a simulation object
local sim_ob = require(opt.sim_file)
-- manipulation_sim{step_interval=1}

-- override print to always flush the output
local old_print = print
local print = function(...)
    old_print(...)
    io.flush()
end

local start_t = os.clock()
for i=1,20 do
  local t0 = os.clock()
  -- TODO Connect to a neural network
  local action = 6
  screen = sim_ob:picking_sim(action)
  print("Frame time-cost: ", os.clock() - t0)
  -- print(screen:size())
  win_input = image.display{image=screen, win=win_input}
  -- image.save('screenshots_' .. i .. '.png', screen)
  print("Step: ",i)
  -- local arm_pose = sim_ob:getArmPose('left')
  -- print(torch.Tensor(arm_pose))
end
local end_t = os.clock()
print("Average frame time-cost: ", (end_t - start_t)/50)


-- sim_ob.vrep:close_vrep()
