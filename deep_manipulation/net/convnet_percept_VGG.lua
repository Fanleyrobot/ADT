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

    local n_hid          = {256,64}
    local n_output       = 3
    -- args.nl             = nn.Rectifier

    local model = {}
    table.insert(model, {'conv1_1', nn.SpatialConvolution(3, 64, 3, 3, 1, 1, 1, 1)})
    table.insert(model, {'relu1_1', nn.ReLU(true)})
    table.insert(model, {'conv1_2', nn.SpatialConvolution(64, 64, 3, 3, 1, 1, 1, 1)})
    table.insert(model, {'relu1_2', nn.ReLU(true)})
    table.insert(model, {'pool1', nn.SpatialMaxPooling(2, 2, 2, 2, 0, 0):ceil()})
    table.insert(model, {'conv2_1', nn.SpatialConvolution(64, 128, 3, 3, 1, 1, 1, 1)})
    table.insert(model, {'relu2_1', nn.ReLU(true)})
    table.insert(model, {'conv2_2', nn.SpatialConvolution(128, 128, 3, 3, 1, 1, 1, 1)})
    table.insert(model, {'relu2_2', nn.ReLU(true)})
    table.insert(model, {'pool2', nn.SpatialMaxPooling(2, 2, 2, 2, 0, 0):ceil()})
    table.insert(model, {'conv3_1', nn.SpatialConvolution(128, 256, 3, 3, 1, 1, 1, 1)})
    table.insert(model, {'relu3_1', nn.ReLU(true)})
    table.insert(model, {'conv3_2', nn.SpatialConvolution(256, 256, 3, 3, 1, 1, 1, 1)})
    table.insert(model, {'relu3_2', nn.ReLU(true)})
    table.insert(model, {'conv3_3', nn.SpatialConvolution(256, 256, 3, 3, 1, 1, 1, 1)})
    table.insert(model, {'relu3_3', nn.ReLU(true)})
    table.insert(model, {'pool3', nn.SpatialMaxPooling(2, 2, 2, 2, 0, 0):ceil()})
    table.insert(model, {'conv4_1', nn.SpatialConvolution(256, 512, 3, 3, 1, 1, 1, 1)})
    table.insert(model, {'relu4_1', nn.ReLU(true)})
    table.insert(model, {'conv4_2', nn.SpatialConvolution(512, 512, 3, 3, 1, 1, 1, 1)})
    table.insert(model, {'relu4_2', nn.ReLU(true)})
    table.insert(model, {'conv4_3', nn.SpatialConvolution(512, 512, 3, 3, 1, 1, 1, 1)})
    table.insert(model, {'relu4_3', nn.ReLU(true)})
    table.insert(model, {'pool4', nn.SpatialMaxPooling(2, 2, 2, 2, 0, 0):ceil()})
    table.insert(model, {'conv5_1', nn.SpatialConvolution(512, 512, 3, 3, 1, 1, 1, 1)})
    table.insert(model, {'relu5_1', nn.ReLU(true)})
    table.insert(model, {'conv5_2', nn.SpatialConvolution(512, 512, 3, 3, 1, 1, 1, 1)})
    table.insert(model, {'relu5_2', nn.ReLU(true)})
    table.insert(model, {'conv5_3', nn.SpatialConvolution(512, 512, 3, 3, 1, 1, 1, 1)})
    table.insert(model, {'relu5_3', nn.ReLU(true)})
    table.insert(model, {'pool5', nn.SpatialMaxPooling(2, 2, 2, 2, 0, 0):ceil()})
    table.insert(model, {'torch_view', nn.View(-1):setNumInputDims(3)})
    table.insert(model, {'fc6', nn.Linear(25088, n_hid[1])})
    table.insert(model, {'relu6', nn.ReLU(true)})
    table.insert(model, {'fc7', nn.Linear(n_hid[1], n_hid[2])})
    table.insert(model, {'relu7', nn.ReLU(true)})
    table.insert(model, {'output', nn.Linear(n_hid[2], n_output)})
    -- table.insert(model, {'prob', nn.SoftMax()})
    args.list_modules = model

    return create_network(args)
end
