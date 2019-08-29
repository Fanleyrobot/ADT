--[[ File
@description:
    This file is to define the architecture of a CNN based on VGG.
@version: V0.00
@author: Fangyi Zhang   email:gzzhangfangyi@gmail.com
@acknowledgement:
    ARC Centre of Excellence for Robotic Vision (ACRV)
    Queensland Univsersity of Technology (QUT)
    Google DeepMind
@history:
    V0.00   05/07/2017  developed the first version
]]

require 'net/list2net'

return function(args)


    -- input 224*224

    local n_hid          = {256,256,256,256,256}
    local n_output       = 2
    -- args.nl             = nn.Rectifier

    local model = {}
    table.insert(model, {'fc8', nn.Linear(256, n_hid[1])})
    table.insert(model, {'relu8', nn.ReLU(true)})
    table.insert(model, {'fc9', nn.Linear(n_hid[1], n_hid[2])})
    table.insert(model, {'relu9', nn.ReLU(true)})
    table.insert(model, {'fc10', nn.Linear(n_hid[2], n_hid[3])})
    table.insert(model, {'relu10', nn.ReLU(true)})
    table.insert(model, {'fc11', nn.Linear(n_hid[3], n_hid[4])})
    table.insert(model, {'relu11', nn.ReLU(true)})
    table.insert(model, {'fc12', nn.Linear(n_hid[4], n_hid[5])})
    table.insert(model, {'relu12', nn.ReLU(true)})
    table.insert(model, {'output', nn.Linear(n_hid[5], n_output)})
    table.insert(model, {'prob', nn.LogSoftMax()})
    -- table.insert(model, {'prob', nn.SoftMax()})
    args.list_modules = model

    return create_network(args)
end
