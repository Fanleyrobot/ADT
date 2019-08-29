--[[
Copyright (c) 2014 Google Inc.

See LICENSE file for full terms of limited license.
]]

require "common/initenv"
require "common/load_pretrained_nets" -- for loading pre-trained networks

function create_network_percep(args)

    local net = nn.Sequential()
    net:add(nn.Reshape(unpack(args.input_dims)))

    --- first convolutional layer
    local convLayer = nn.SpatialConvolution

    -- Comment the following codes, it is just for legacy purposes.
    -- SpatialConvolution is only one class that is the gold standard for convolutions. It works both on CPU and CUDA efficiently.
    -- Cuda-convnet2 is a special case because it only supports square kernels, and special cases of those.
    -- Different classes have different initial parameters.

    --if args.gpu >= 0 then
    --    net:add(nn.Transpose({1,2},{2,3},{3,4}))
    --    convLayer = nn.SpatialConvolutionCUDA
    --end

    net:add(convLayer(args.hist_len*args.ncols, args.n_units[1],
                        args.filter_size[1], args.filter_size[1],
                        args.filter_stride[1], args.filter_stride[1],1)) -- add 1 zero in each width and height on the input planes
    net:add(args.nl())

    -- Add convolutional layers
    for i=1,(#args.n_units-1) do
        -- second convolutional layer
        net:add(convLayer(args.n_units[i], args.n_units[i+1],
                            args.filter_size[i+1], args.filter_size[i+1],
                            args.filter_stride[i+1], args.filter_stride[i+1]))
        net:add(args.nl())
    end

    local nel
    if args.gpu >= 0 then
        --net:add(nn.Transpose({4,3},{3,2},{2,1})) -- comment this line, since it is useless and just for legacy purposes.
        nel = net:cuda():forward(torch.zeros(1,unpack(args.input_dims))
                :cuda()):nElement()
    else
        nel = net:forward(torch.zeros(1,unpack(args.input_dims))):nElement()
    end

    -- reshape all feature planes into a vector per example
    net:add(nn.Reshape(nel))

    -- fully connected layer
    -- net:add(nn.Linear(nel, args.n_hid[1]))
    -- net:add(args.nl())
    local last_layer_size = args.n_hid[1]

    for i=1,(#args.n_hid-1) do
        -- add Linear layer
        last_layer_size = args.n_hid[i+1]
        net:add(nn.Linear(args.n_hid[i], last_layer_size))
        net:add(args.nl())
    end

    -- -- add the last fully connected layer (to actions)
    -- net:add(nn.Linear(last_layer_size, args.n_actions))

    if args.gpu >=0 then
        net:cuda()
    end
    if args.verbose >= 2 then
        print(net)
        print('Convolutional layers flattened output size:', nel)
    end
    return net
end

function create_network_ctrl(args)

    local net = nn.Sequential()

    local nel = args.state_dim*args.hist_len*args.ncols -- the vector dimension of the input layer

    -- reshape all feature planes into a vector per example
    net:add(nn.Reshape(nel))

    -- fully connected layer
    net:add(nn.Linear(nel, args.n_hid[1]))
    net:add(args.nl())
    local last_layer_size = args.n_hid[1]

    for i=1,(#args.n_hid-1) do
        -- add Linear layer
        last_layer_size = args.n_hid[i+1]
        net:add(nn.Linear(args.n_hid[i], last_layer_size))
        net:add(args.nl())
    end

    -- add the last fully connected layer (to actions)
    net:add(nn.Linear(last_layer_size, args.n_actions))

    if args.gpu >=0 then
        net:cuda()
    end
    if args.verbose >= 2 then
        print(net)
    end
    return net
end

function create_network(args)
  args.verbose = 2
  args.hist_len = 1
  args.ncols = 3
  args.gpu = -1
  args.input_dims     = {3, 129, 129}
  args.n_units        = {64, 64, 64}
  args.filter_size    = {7, 4, 3}
  args.filter_stride  = {4, 2, 1}
  -- args.n_hid          = {4, 400,300}  -- 2dof
  args.n_hid          = {3}  -- x y z
  args.nl             = nn.Rectifier
  -- local net_percept = create_network_percep(args)

  -- args.state_dim = 10
  args.state_dim = 10816 + 7
  args.ncols = 1
  args.n_hid          = {400,300}
  args.n_actions      = 7
  -- args.nl             = nn.Rectifier

  -- local net_ctrl = create_network_ctrl(args)

  -- local net_percept = load_pretrained_network("CNN_P001_0_n_u_3_G_01_129_580000.t7")
  -- local net_percept = load_pretrained_network("CNN_P0_256_v8_FT_14_1.0_45000.t7")
  -- local net_ctrl = load_pretrained_network("FC_7DVel_SL_003_0_6000000.t7")
  -- local net_ctrl = load_pretrained_network("FC_7DVel_SL_005_0_11000000.t7")

  local net_percept = load_pretrained_network("CNN_conv_for_L.t7")
  local net_ctrl = load_pretrained_network("FC_fc_for_L.t7")

  -- local net_ctrl = load_pretrained_network("FC_7DVel_SL_005_01_5000000.t7")
  local net_combined = nn.Sequential()
  local net_temp = nn.ParallelTable()
  net_temp:add(nn.Copy())
  net_temp:add(net_percept)
  net_combined:add(net_temp)
  -- net_combined:add(nn.JoinTable(1))
  -- the second parameter tells the number of dimentions for one input
  -- which makes it compatible to both a single frame and minibatch
  net_combined:add(nn.JoinTable(1,1))
  net_combined:add(net_ctrl)
  print(net_combined)

  filename = "complex_net_256_v8_P0_L"
  -- torch.save(filename .. ".t7", {model=net_combined:clearState():float(),
  --                               p_model = net_percept:clone():clearState():float()})
  torch.save(filename .. ".t7", {model=net_combined:clearState():float()})
  print('Saved:', filename .. '.t7')
  io.flush()
  collectgarbage()
end

create_network({})
