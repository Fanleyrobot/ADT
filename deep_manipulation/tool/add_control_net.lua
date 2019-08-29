--[[ File
@description:
    This file is for merging perception and control networks for semi-supervised end-to-end fine-tuning.
@version: V0.00
@author: Fangyi Zhang   email:gzzhangfangyi@gmail.com
@acknowledgement:
    ARC Centre of Excellence for Robotic Vision (ACRV)
    Queensland Univsersity of Technology (QUT)
@history:
    V0.00   21/04/2018  developed the first version
]]

if not dmn then
    require "common/initenv"
end
require 'common/load_pretrained_nets'

-- get settings from the running script
local cmd = torch.CmdLine()
cmd:text()
cmd:text('To merge the perception and control networks ...')
cmd:text()
cmd:text('Options:')

-- Added parameters
cmd:option('-pnetwork', 'perception_network', 'the t7 file of the perception network, e.g., perception_network.t7') --Set simulator filename
cmd:option('-cnetwork', 'control_network', 'the t7 file of the control network, e.g., ctrl_network.t7') --Set simulator filename
cmd:option('-e2e_net_name', 'e2e_network', 'the name of a t7 file to save the merged network, e.g., e2e_network.t7') --Set simulator filename


-- cmd:option('-index', 'p_model', 'the index of the perception network in the end-to-end network, e.g., p_model') --Set simulator filename
-- End Addition

cmd:text()

local opt = cmd:parse(arg)

-- override print to always flush the output
local old_print = print
local print = function(...)
    old_print(...)
    io.flush()
end

local p_group = torch.load(opt.pnetwork)

local encoder_sr = load_pretrained_network(opt.pnetwork, 'encoder_sr')
local encoder_tg = load_pretrained_network(opt.pnetwork, 'encoder_tg')
local d_net = load_pretrained_network(opt.pnetwork, 'd_net')
local pose_net = load_pretrained_network(opt.pnetwork, 'pose_net')
local ctrl_net = load_pretrained_network(opt.cnetwork)

print('Source Encoder Net: ')
print(encoder_sr)
print('Target Encoder Net: ')
print(encoder_tg)
print('Pose Regressor Net: ')
print(pose_net)
print('Discriminater Net: ')
print(d_net)
print('Control Net: ')
print(ctrl_net)

torch.save(opt.e2e_net_name, {encoder_tg = encoder_tg:clearState():float(),
                        encoder_sr = encoder_sr:clearState():float(),
                        d_net = d_net:clearState():float(),
                        pose_net = pose_net:clearState():float(),
                      ctrl_net = ctrl_net:clearState():float()})
print("Perception and Control networks merged!")
print('Saved as:',opt.e2e_net_name)
io.flush()
collectgarbage()