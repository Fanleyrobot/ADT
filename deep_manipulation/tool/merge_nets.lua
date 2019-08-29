--[[ File
@description:
    This file is for generating a network initialized with pre-trained weights.
@version: V0.12
@author: Fangyi Zhang   email:gzzhangfangyi@gmail.com
@acknowledgement:
    ARC Centre of Excellence for Robotic Vision (ACRV)
    Queensland Univsersity of Technology (QUT)
@history:
    V0.00   12/05/2016  developed the first version
    V0.10   17/05/2016  added the function of loading pre-trained percepiton networks
    V0.11   17/06/2016  made it compatible to caffe and torch models
    V0.12   07/03/2017  added the functionality of partially initializing a certain layer
]]

-- detect whether "dmn" has been constructed or is still nil
if not dmn then
    require "common/initenv"
end
require "common/network_visualization" -- for network visualization
require "common/load_pretrained_nets" -- for loading pre-trained networks
require "common/network_operation"
require 'image'

-- get settings from the running script
local cmd = torch.CmdLine()
cmd:text()
cmd:text('Generate a network and partly initialize it with pre-trained networks (Conv or FC layers):')
cmd:text()
cmd:text('Options:')

cmd:option('-name', '', 'filename used for saving the generated network')
cmd:option('-network', '', 'reload pretrained network')
cmd:option('-cnetwork', 'nil', 'reload pretrained control network')
cmd:option('-pnetwork', 'nil', 'reload pretrained perception network')
cmd:option('-pose_netfile', 'nil', 'pose network file')
cmd:option('-d_netfile', 'nil', 'discriminator network file')
cmd:option('-caffe_prototxt', 'nil', 'the full directory of the deploy.prototxt file of a pretrained model')
cmd:option('-caffe_binary', 'nil', 'the full directory of the *.caffemodel file of a pretrained model')
cmd:option('-agent', '', 'name of agent file to use')
cmd:option('-agent_params', '', 'string of agent parameters')
-- cmd:option('-seed', 1, 'fixed input seed for repeatable experiments')

-- Layer matching between the network to be initialized and the pretrained networks
cmd:option('-percept_layers', '{}', 'the index of perception layers to be initialized')
cmd:option('-con_layers', '{}', 'the index of control layers to be initialized')
cmd:option('-pre_percept_layers', '{}', 'the corresponding index of layers in a pre-trained perception network')
cmd:option('-pre_con_layers', '{}', 'the corresponding index of layers in a pre-trained control network')

cmd:option('-verbose', 2, 'the higher the level, the more information is printed to screen')
cmd:option('-threads', 1, 'number of BLAS threads')
cmd:option('-gpu', -1, 'gpu flag')

cmd:text()

local opt = cmd:parse(arg)

--- Generate a network with random initialization
local game_actions, agent, opt = setup(opt)


-- local pose_netfile="net/fcnet_percept_VGG_smaller_v8_Pose"
-- local d_netfile="net/fcnet_percept_VGG_smaller_v8_Discriminater17"



-- local pose_netfile="net/fcnet_percept_VGG_smaller_v8_Pose_with_shortcut"
-- local d_netfile="net/fcnet_percept_VGG_smaller_v8_Discriminater_with_shortcut3_1"
-- local d_netfile="net/convnet_percept_VGG_smaller_v8_Discriminater"

local args = {}
args.gpu = -1
args.verbose = 2

local pose_netfile=opt.pose_netfile
local d_netfile=opt.d_netfile
local msg, err = pcall(require, d_netfile)
print('Creating Agent Network from ' .. d_netfile)
local d_net = err(args)

msg, err = pcall(require, pose_netfile)
print('Creating Agent Network from ' .. pose_netfile)
local pose_net = err(args)

--
-- -- Initialize a visulization object for the network
-- local input_dims = agent.input_dims
-- local visu_ob = network_visualization{network=agent.network,
--                                         network_type=opt.gpu,
--                                         input_dims=input_dims}

-- override print to always flush the output
local old_print = print
local print = function(...)
    old_print(...)
    io.flush()
end


-- Visualize the network with random initializations
-- local merged_img_weight, weight_group, weight_sub_group = visu_ob:visualize_weights()
-- image.save("random_weights.png", merged_img_weight)

-- Define the layer index
-- local caffe_model
local pre_trained_per
local pre_trained_con

-- local percept_layers = {1,5}
-- local pre_percept_layers = {1,4}


local percept_layers = {1,4,7,9,12,14,17,19,22,24,27,29}
local pre_percept_layers = {1,4,7,9,12,14,17,19,22,24,27,29}

-- local percept_layers = opt.percept_layers
-- local pre_percept_layers = opt.pre_percept_layers

-- local percept_layers = {1,4,7,10,13,17,19,21}
-- local pre_percept_layers = {1,6,11,18,25,33,35,37}
-- local percept_layers = {7,10,13,17,19,21}
-- local pre_percept_layers = {11,18,25,33,35,37}
-- local percept_layers = {1}
-- local pre_percept_layers = {1}
-- local con_layers = {1} -- the index of layers for control
-- local pre_con_layers = {1} -- the index of weighted layers in a pre-trained control network

-- -- Load a pre-trained caffe model, using it as a pre-trained perception network
-- pre_trained_per = Caffe2Torch(opt.caffe_prototxt,opt.caffe_binary)

-- Load a pre-trained perception network
print("Pretrained model loading ....")
if opt.pnetwork~='nil' then
    pre_trained_per = load_pretrained_network(opt.pnetwork)
    if opt.gpu and opt.gpu >= 0 then
        pre_trained_per:cuda()
    else
        pre_trained_per:float()
    end
end
--
-- print("Pretrained model loaded ....")
-- -- Load a pre-trained control network
-- if opt.cnetwork~='nil' then
--     pre_trained_con = load_pretrained_network(opt.cnetwork)
-- end

-- Initialize the perception part
local m = #percept_layers
for i=1,m do
    -- local r = pre_trained_per:get(pre_percept_layers[i]).weight:select(2, 3)
    -- local g = pre_trained_per:get(pre_percept_layers[i]).weight:select(2, 2)
    -- local b = pre_trained_per:get(pre_percept_layers[i]).weight:select(2, 1)
    -- -- print(pre_trained_per:get(pre_percept_layers[i]).bias)
    -- -- local sum = 0.2989*r + 0.5870*g + 0.1140*b
    -- agent.network:get(percept_layers[i]).weight:select(2,1):copy(r)
    -- agent.network:get(percept_layers[i]).weight:select(2,2):copy(g)
    -- agent.network:get(percept_layers[i]).weight:select(2,3):copy(b)


    -- local w = pre_trained_per:get(i).weight
    -- local b = pre_trained_per:get(i).bias
    -- if w then
    --   if i==1 then
    --     print(w:size())
    --     w=w:index(2,torch.LongTensor{3,2,1})
    --     -- w=w:mul(255)
    --   end
    print("Layer: ", i)
    --
    --   agent.network:get(i).weight:copy(w)
    --   agent.network:get(i).bias:copy(b)
    -- end
    agent.network:get(percept_layers[i]).weight:copy(pre_trained_per:get(pre_percept_layers[i]).weight)
    agent.network:get(percept_layers[i]).bias:copy(pre_trained_per:get(pre_percept_layers[i]).bias)

end


-- -- Initialize the pose net
-- local pose_layers = {1,3,5}
-- local pre_pose_layers = {33,35,37}
-- m = #pose_layers
-- for i=1,m do
--     -- local r = pre_trained_per:get(pre_percept_layers[i]).weight:select(2, 3)
--     -- local g = pre_trained_per:get(pre_percept_layers[i]).weight:select(2, 2)
--     -- local b = pre_trained_per:get(pre_percept_layers[i]).weight:select(2, 1)
--     -- -- print(pre_trained_per:get(pre_percept_layers[i]).bias)
--     -- -- local sum = 0.2989*r + 0.5870*g + 0.1140*b
--     -- agent.network:get(percept_layers[i]).weight:select(2,1):copy(r)
--     -- agent.network:get(percept_layers[i]).weight:select(2,2):copy(g)
--     -- agent.network:get(percept_layers[i]).weight:select(2,3):copy(b)
--
--
--     -- local w = pre_trained_per:get(i).weight
--     -- local b = pre_trained_per:get(i).bias
--     -- if w then
--     --   if i==1 then
--     --     print(w:size())
--     --     w=w:index(2,torch.LongTensor{3,2,1})
--     --     -- w=w:mul(255)
--     --   end
--     print("Layer: ", i)
--     --
--     --   agent.network:get(i).weight:copy(w)
--     --   agent.network:get(i).bias:copy(b)
--     -- end
--     pose_net:get(pose_layers[i]).weight:copy(pre_trained_per:get(pre_pose_layers[i]).weight)
--     pose_net:get(pose_layers[i]).bias:copy(pre_trained_per:get(pre_pose_layers[i]).bias)
--
-- end


-- A new way of copying weights
local p_w, p_dw = pose_net:getParameters()
local n_p_w = p_w:size(1)
local w, dw = pre_trained_per:getParameters()
local n_w = w:size(1)
print(n_p_w)
print(n_w)
local sub_w = w:narrow(1,n_w-n_p_w+1,n_p_w)
print(p_w[200])
p_w:copy(sub_w)
print(p_w[200])
print(sub_w[200])

-- con1_size_w = pre_trained_con:get(2).weight:size()
-- con1_size_b = pre_trained_con:get(2).bias:size()
-- c1_size_w = agent.network:get(10).weight:size()
-- c1_size_b = agent.network:get(10).bias:size()
-- print("con1_size_w:", con1_size_w)
-- print("c1_size_w:", c1_size_w)
-- print("con1_size_b:", con1_size_b)
-- print("c1_size_b:", c1_size_b)

-- -- Initialize the intermediate layer
-- agent.network:get(10).weight:narrow(2,1,5):copy(pre_trained_con:get(2).weight)
-- agent.network:get(10).bias:copy(pre_trained_con:get(2).bias)

-- -- Initialize the control part
-- m = #con_layers
-- for i=1,m do
--     print("Layer: ", i)
--     agent.network:get(con_layers[i]).weight:copy(pre_trained_con:get(pre_con_layers[i]).weight)
--     agent.network:get(con_layers[i]).bias:copy(pre_trained_con:get(pre_con_layers[i]).bias)
-- end

-- Visualize the network initialized with pre-trained networks
-- merged_img_weight, weight_group, weight_sub_group = visu_ob:visualize_weights()
-- image.save("initialized_weights.png", merged_img_weight)

-- print("sample weights:", agent.network:get(9).weight:narrow(2,1,5))

-- -- Construct a perception network for the fine-tuning with weighted cost functions
-- p_model = load_pretrained_network("CNN_GNetFinetune13_2300000.t7")
-- if opt.gpu and opt.gpu >= 0 then
--     p_model:cuda()
-- else
--     p_model:float()
-- end
-- -- Update p_model
-- m = #percept_layers
-- for i=1,m-1 do
--     -- local r = pre_trained_per:get(pre_percept_layers[i]).weight:select(2, 3)
--     -- local g = pre_trained_per:get(pre_percept_layers[i]).weight:select(2, 2)
--     -- local b = pre_trained_per:get(pre_percept_layers[i]).weight:select(2, 1)
--     -- local sum = 0.2989*r + 0.5870*g + 0.1140*b
--     -- agent.network:get(percept_layers[i]).weight:copy(sum)
--     p_model:get(percept_layers[i]).weight:copy(pre_trained_per:get(pre_percept_layers[i]).weight)
--     p_model:get(percept_layers[i]).bias:copy(pre_trained_per:get(pre_percept_layers[i]).bias)
-- end
-- -- Initialize the last perception layer
-- p_model:get(9).weight:copy(pre_trained_per:get(9).weight:narrow(1,1,5))
-- p_model:get(9).bias:copy(pre_trained_per:get(9).bias:narrow(1,1,5))
-- local visu_ob_p = network_visualization{network=p_model,
--                                         network_type=opt.cpu,
--                                         input_dims=input_dims}
-- merged_img_weight, weight_group, weight_sub_group = visu_ob_p:visualize_weights()
-- image.save("initialized_weights_p.png", merged_img_weight)
-- local p_model = extractPnet(opt.e2e_net)

local encoder_sr = agent.network:clearState():clone():float()
local encoder_tg = encoder_sr:clone()

-- save the CPU network
local filename = opt.name
-- torch.save(filename .. ".t7", {model=agent.network:clearState():float(),
--                               p_model = p_model:clearState():float()})
-- torch.save(filename .. ".t7", {model=agent.network:clearState():float()})
-- torch.save(filename .. ".t7", {model = pre_trained_per:clearState():float(),
--                               auto_model=agent.network:clearState():float()})

torch.save(filename .. ".t7", {encoder_tg = encoder_tg,
                        encoder_sr = encoder_sr,
                        d_net = d_net,
                        pose_net = pose_net})

print('Saved:', filename .. '.t7')
io.flush()
collectgarbage()
