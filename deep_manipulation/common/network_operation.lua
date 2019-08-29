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

-- detect whether "dqn" has been constructed or is still nil
if not dqn then
    require "common/initenv"
end
require "common/load_pretrained_nets" -- for loading pre-trained networks

-- Load a caffe model and save as a torch network in a t7 file
function Caffe2Torch(caffe_prototxt,caffe_binary)
  if caffe_binary == 'nil' then
    print("No caffe binary file!!!")
  elseif caffe_prototxt == 'nil' then
    print("No caffe protocal file!!!")
  else
    local caffe_model = load_caffe_model(caffe_prototxt,caffe_binary)
    return caffe_model
  end

end

-- Copy weights from a source network
function copyW_layer(target_net, source_net, target_layers, source_layers)
  local m = #target_layers
  local n = #source_layers
  if m ~= n then
    print("The number of source and target layers should be consistent!!!")
  else
    for i=1,m do
      target_net:get(target_layers[i]).weight:copy(source_net:get(source_layers[i]).weight)
      target_net:get(target_layers[i]).bias:copy(source_net:get(source_layers[i]).bias)
    end
  end
end

-- Copy weights in the vector manner
function copyW_vector(target_net,source_net,target_range, source_range)
  local source_w, source_dw = source_net:getParameters()
  local w, dw = target_net:getParameters()
  if not target_range or not source_range then
    print("Copy all the weights in the source net to the target net from the starting index!!!")
    local n_source_w = source_w:size(1)
    local sub_w = w:narrow(1,1,n_source_w)
    sub_w:copy(source_w)
  else
    print("Copy weights according to the given ranges!!!")
    local source_sub_w = source_w:narrow(1,source_range[1],source_range[2])
    local sub_w = w:narrow(1,target_range[1],target_range[2])
    sub_w:copy(source_sub_w)
  end
  return target_net
end

-- Get the weightable index of a network
function getWeightableIndex(net)
  local m = #net
  local w_index = {}
  for i=1,m do
    if net:get(i).weight then
      w_index[#w_index+1] = i
    end
  end
  print("Weightabl index: ", unpack(w_index))
  return w_index
end

-- Convert BGR weights to RGB weights
function BGR2RGB(w, dim)
  if not dim then
    dim = 2
  end
  return w:index(dim,torch.LongTensor{3,2,1})
end

-- Convert RGB weights to Grey-scale weights
function RGB2Grey(w)
  local r = w:select(2, 1)
  local g = w:select(2, 2)
  local b = w:select(2, 3)
  -- print(pre_trained_per:get(pre_percept_layers[i]).bias)
  local sum = 0.2989*r + 0.5870*g + 0.1140*b
  return sum
end

-- Extract the perception network in a end-to-end network
function extractPnet(net)
  return load_pretrained_network(net,'p_model')
end
