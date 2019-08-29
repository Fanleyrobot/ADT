--[[ File
@description:
    This file is for extracting a perception network in an end-to-end network.
@version: V0.00
@author: Fangyi Zhang   email:gzzhangfangyi@gmail.com
@acknowledgement:
    ARC Centre of Excellence for Robotic Vision (ACRV)
    Queensland Univsersity of Technology (QUT)
@history:
    V0.00   23/08/2017  developed the first version
]]

if not dmn then
    require "common/initenv"
end
require 'common/load_pretrained_nets'

-- get settings from the running script
local cmd = torch.CmdLine()
cmd:text()
cmd:text('To extract the perception network in an end-to-end network ...')
cmd:text()
cmd:text('Options:')

-- Added parameters
cmd:option('-network', 'e2e_network', 'the t7 file of the end-to-end network, e.g., e2e_network.t7') --Set simulator filename
cmd:option('-index', 'p_model', 'the index of the perception network in the end-to-end network, e.g., p_model') --Set simulator filename
-- End Addition

cmd:text()

local opt = cmd:parse(arg)

-- override print to always flush the output
local old_print = print
local print = function(...)
    old_print(...)
    io.flush()
end

local p_net = load_pretrained_network(opt.network, opt.index)
print('The perception network architecture: ')
print(p_net)

filename = opt.index .. '_from_' .. opt.network
torch.save(filename, {model=p_net:clearState():float()})
print("The perception network has been extracted successfully!")
print('Saved as:', filename)
io.flush()
collectgarbage()