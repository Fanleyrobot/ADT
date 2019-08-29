--[[ File
@description:
    This file is for the pre-trained network transformation from GPU to CPU.
@version: V0.10
@author: Fangyi Zhang   email:gzzhangfangyi@gmail.com
@acknowledgement:
    ARC Centre of Excellence for Robotic Vision (ACRV)
    Queensland Univsersity of Technology (QUT)
@history:
    V0.00   06/08/2015  developed the first version
    V0.10   26/07/2018  minor updates for code re-organization
]]

-- detect whether "dmn" has been constructed or is still nil
if not dmn then
    require "common/initenv"
end
require "common/utilities" -- some utilization functions

--require 'cutorch'
require 'cunn'

-- get settings from the running script
local cmd = torch.CmdLine()
cmd:text()
cmd:text('Transform a pre-trained network from GPU to CPU:')
cmd:text()
cmd:text('Options:')

cmd:option('-name', '', 'filename used for saving a transferred network')
cmd:option('-network', '', 'the pretrained network to load')
cmd:option('-model_index', '', 'the index of the model to be converted')


cmd:text()

local opt = cmd:parse(arg)

-- override print to always flush the output
local old_print = print
local print = function(...)
    old_print(...)
    io.flush()
end

local t7_file = load_t7_File(opt.network)
local model_index = opt.model_index or "model"
local cuda_model = t7_file[model_index]
local cpu_model = CudaNN_To_CpuNN(cuda_model)

-- save the CPU network
local filename = opt.name
torch.save(filename .. ".t7", {model = cpu_model})
                        -- best_model = cpu_best_model})
print('Saved:', filename .. '.t7')
io.flush()
collectgarbage()