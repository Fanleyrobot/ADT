--[[ File
@description:
    The functions included in this file are for network visualization.
@version: V0.50
@author: Fangyi Zhang   email:gzzhangfangyi@gmail.com
@acknowledgement:
    ARC Centre of Excellence for Robotic Vision (ACRV)
    Queensland Univsersity of Technology (QUT)
@history:
    V0.00   31/07/2015  developed the first version
    V0.10   01/08/2015  added the generation of a merged image
    V0.11   03/08/2015  updated pixel grey scale values normalization codes
    V0.20   03/08/2015  integrated into a class
    V0.22   03/08/2015  added the image substraction function
    V0.23   04/08/2015  updated the image substraction function to absolute substraction
    V0.24   06/08/2015  updated to support GPU network
    V0.25   07/08/2015  renamed to network_visualization
    V0.26   07/08/2015  updated to ensure the width of the merged image is divisible by 2
    V0.30   17/11/2015  added two more visualization color_mode: "heatmap" and "three_color"
    V0.35   17/11/2015  added three different data normalization modes: "zero_centric1", "zero_centric2", and "min_max"
    V0.36   17/11/2015  updated the substraction_image function
    V0.38   18/11/2015  added the verticle image merging mode
    V0.39   18/11/2015  added a new normalization mode: zero_centric3
    V0.40   18/11/2015  added more visualization mode: "output", "weight", "substraction", and their combinations.
    V0.41   19/11/2015  fixed a minor bug (tensor dimension conflict which happens in the training) in the function of visualize_tensor and visualize_single_layer_output
    V0.42   12/12/2015  added weights visualization
    V0.43   14/12/2015  fixed the bug of not being divisible by 2
    V0.44   12/05/2016  fixed the bug of initializing the visu_output and visu_weight with default values
    V0.45   16/05/2016  fixed the bug of incomplete weight visualization
    V0.46   19/05/2016  added the function of autonomously initialize simage_h, simage_w and w_layers.
    V0.48   20/05/2016  added the separated functions for outputs and weights visualization
    V0.49   21/05/2016  fixed the bug of temp state type conflict for a cuda network
    V0.50   02/06/2016  optimized the code to reduce memory usage
]]

-- TODO:
-- 1.Add a function to highlight the neuron with high impact and weaken those with low impact
-- 2.The impact can be determined by the value itself, or through combining with the weight for that unit in the following layer.

require 'torch'
require 'image'
require 'math'
--require "mattorch"

-- construct a class
local visu = torch.class('network_visualization')


--[[ Function
@description: initialize a network visualization object
@input:
    args: settings for a network visualization object, i.e., {network=agent.network}
@output: nil
@notes: the network is the only argument that has to be set at least.
]]
function visu:__init(args)

    self.network = args.network -- a network container
    self.network_type = args.network_type or -1 -- the type of the network: float or cuda
    self.input_dims = args.input_dims -- Set input dims
    self.n_modules = self.network:size() -- number of layers (modules included in a network container)

    self.simage_h, self.simage_w, self.w_layers = self:get_WeightableLayers_and_OutputSizes()
    -- print("simage_h : ", self.simage_h)
    -- print("simage_w : ", self.simage_w)
    -- print("w_layers : ", self.w_layers)

    -- Settings for outputs visualization
    -- image size definition for the transformation from vectors to images
    -- TODO: adapt the image sizes when initializing a new object
    -- self.simage_h = args.simage_h or {0, 0, 0, 0, 0, 0, 0, 28, 16, 16, 9}
    -- self.simage_w = args.simage_w or {0, 0, 0, 0, 0, 0, 0, 28, 8, 8, 1}
    -- -- self.simage_h = args.simage_h or {3, 16, 16, 9}
    -- -- self.simage_w = args.simage_w or {1, 8, 8, 1}

    -- -- Settings for weights visualization
    -- -- Set the layers for weight visualization
    -- self.w_layers = args.w_layers or {2, 4, 6, 9, 11}
    -- self.w_layers = args.w_layers or {2, 4}
    -- image size definition for the transformation from vectors to images
    -- self.simage_h = args.simage_h or {0, 0, 0, 0, 0}
    -- self.simage_w = args.simage_w or {0, 0, 0, 0, 0}

    -- settings for the visualized image color mode: "heatmap", "three_color", "mono"
    self.color_mode = args.color_mode or "heatmap"

    -- settings for the data normalization mode: "zero_centric1", "zero_centric2", "zero_centric3", "min_max"
    -- zero_centric1: normalize positive and negative data independently with a consitent ratio, without any offsets
    -- zero_centric2: normalize positive and negative data independently with different ratios and offsets
    -- zero_centric3: normalize positive and negative data independently with different ratios
    -- min_max: normalize the data according to min and max values, -1~1
    self.norm_mode = args.norm_mode or "min_max"

    -- settings for visualization mode:
    -- When initializing the boolean variables, the default values should be set to false,
    -- otherwise no matter what value in the args, it will be set to true,
    -- due to the grammer of args.visu_* or *, when args.visu_* is nil or false,
    -- the self.visu_* will be initialized using default values.
    -- self.visu_output = args.visu_output or false -- whether visualize outputs
    -- self.visu_weight = args.visu_weight or false -- whether visualize weights
    self.visu_output_sub = args.visu_output_sub or false -- whether visualize the frame-by-frame substraction
    self.visu_weight_sub = args.visu_weight_sub or false -- whether visualize the frame-by-frame substraction

    -- settings for saving previous frames
    self.pre_output = args.pre_output or nil -- previous output frame
    self.pre_weight = args.pre_weight or nil -- previous weight frame
    self.pre_interval = args.pre_interval or 1 -- how many interval frames to update the previous frame
    self.pre_interval_count = 0 -- the counting variable for interval frames
    self.pre_interval_count_w = 0 -- the counting variable for interval frames for weights

    -- settings for a merged image
    self.padding_merged_image = args.padding_merged_image or 2

    -- initialize the padding mask for a merged image
    -- self.padding_mask = torch.Tensor(self.padding_merged_image,self.padding_merged_image):fill(1)
    -- for i=1,self.n_modules do
    --     local imgDisplay, restored_output = self:visualize_single_layer_output(i)
    --     -- merge images
    --     imgDisplay:zero()
    --     self.padding_mask = self:merge_image(self.padding_mask, imgDisplay)
    -- end
end

--[[ Function
@description: Get the image dims for a single vector
@input:
    num: a natural number
@output:
    h: hight
    w: width
@notes:
]]
function visu:get_single_output_size(num)
    local w = math.sqrt(num)
    local h = w

    local integer, fraction = math.modf(w)
    if fraction ~= 0 then
        for i = 1, integer do
            if num % i == 0 then
                w = i
            end
        end
        h = num / w
    end

    return h, w
end


--[[ Function
@description: Get the index of weightable layers and the image dim for output visualization
@input:
@output:
    simage_h: the table of hight information for each layer
    simage_w: the table of width information for each layer
    w_layers: the table of the index of each weightable layer
@notes:
    To successfully run this, the parameter input_dims needs to be set.
]]
function visu:get_WeightableLayers_and_OutputSizes()
    local simage_h = {}
    local simage_w = {}
    local w_layers = {}

    -- Forward the fresh network once to initialize some output tensors
    -- In case that the outputs in some layers have not been initialized yet.
    local temp_state = torch.Tensor(1,unpack(self.input_dims))
    if self.network_type >= 0 then -- if the network type is cuda, then convert the temp_state to a cuda tensor
        temp_state = temp_state:cuda()
    end
    self.network:forward(temp_state)

    for i=1,self.n_modules do
        -- Get the current module
        local current_module = self.network:get(i)

        -- Set the output visualization size of each layer
        local original_output = torch.Tensor(current_module.output:float())
        local dim = original_output:dim()
        if dim > 2 then
            simage_h[i] = 0
            simage_w[i] = 0
        else
            if dim > 1 then
                original_output = original_output[1]
            end
            local num = original_output:size(1)
            local h, w = self:get_single_output_size(num)
            simage_h[i] = h
            simage_w[i] = w
        end

        -- Get the index of all weightable layers
        local weights, grad_weights = current_module:parameters()
        if weights then
            w_layers[#w_layers+1] = i
            -- print("weightable layer: ", i)
        end
        -- print("output h, w: ", simage_h[i], simage_w[i])

    end

    return simage_h, simage_w, w_layers

end


--[[ Function
@description: Normalize data
@input:
    origin: the original tensor to be normalized
@output:
    normalized_tensor: the normalized tensor
@notes:
    -- zero_centric1: normalize positive and negative data independently with a consitent ratio, without any offsets
    -- zero_centric2: normalize positive and negative data independently with different ratios and offsets
    -- zero_centric3: normalize positive and negative data independently with different ratios
    -- min_max: normalize the data according to min and max values, to -1~1
]]
function visu:NormalizeTensor(origin)
    local normalized_data

    if self.norm_mode == "min_max" then
        local min_all = torch.min(origin)
        local max_all = torch.max(origin)
        local range = max_all-min_all
        local offset = -min_all

        -- If all the data are the same, then the normalized data would all be zeros, ones or minus ones.
        if range == 0 or range == -0 then
            range = 1
            offset = 0.5
            if max_all > 0 then
                range = max_all
                offset = 0
            elseif min_all < 0 then
                range = 1
                offset = -min_all
            end
        end
        normalized_data = origin:clone():add(offset):div(range):mul(2):add(-1)

    else
        local origin_size = origin:size()
        -- Separate positive and negative data
        local positive_data = torch.Tensor(origin_size):copy(origin):gt(0):float():cmul(origin)
        local negative_data = torch.Tensor(origin_size):copy(origin):lt(0):float():cmul(origin)
        local positive_data_normalized = positive_data:clone()
        local negative_data_normalized = negative_data:clone()
        -- print("red min and max:", torch.min(color_r), ";", torch.max(color_r))
        -- print("blue min and max:", torch.min(color_b), ";", torch.max(color_b))

        local min_positive = torch.min(positive_data)
        local max_positive = torch.max(positive_data)
        local min_negative = torch.min(negative_data)
        local max_negative = torch.max(negative_data)

        if self.norm_mode == "zero_centric1" then
            local data_max = math.max(max_positive,-min_negative)
            -- If all the data are all zeros, then the normalized data would still be zeros.
            if data_max > 0 then
                positive_data_normalized:div(data_max)
                negative_data_normalized:div(data_max)
            end

        elseif self.norm_mode == "zero_centric2" then
            local range = max_positive-min_positive
            local offset = -min_positive
            -- If all the data are the same, then the normalized data would all be zeros or ones.
            if range == 0 or range == -0 then
                range = 1
                if min_positive > 0 then
                    range = min_positive
                    offset = 0
                end
            end
            positive_data_normalized:add(offset):div(range)

            -- If all the data are the same, then the normalized data would all be zeros or minus ones.
            range = max_negative-min_negative
            offset = -max_negative
            if range == 0 or range == -0 then
                range = 1
                if max_negative < 0 then
                    range = -max_negative
                    offset = 0
                end
            end
            negative_data_normalized:add(offset):div(range)

        elseif self.norm_mode == "zero_centric3" then
            local range = max_positive-min_positive
            -- If all the data are the same, then the normalized data would all be zeros or ones.
            if range == 0 or range == -0 then
                range = 1
                if min_positive > 0 then
                    range = min_positive
                end
            end
            positive_data_normalized:div(range)

            -- If all the data are the same, then the normalized data would all be zeros or minus ones.
            range = max_negative-min_negative
            if range == 0 or range == -0 then
                range = 1
                if max_negative < 0 then
                    range = -max_negative
                end
            end
            negative_data_normalized:div(range)
        end

        normalized_data = torch.add(positive_data_normalized, negative_data_normalized)

    end

    return normalized_data
end


--[[ Function
@description: Transform grey-scale map to 3 color map
    Red: 1; Black: 0; Blue: -1
@input:
    origin: the original tensor to be visualized
@output:
    color_tensor: the colorized tensor transformed from the original tensor
@notes:
]]
function visu:GreyToRedBlueMap(origin)
    -- Normalize the original data
    origin = self:NormalizeTensor(origin)
    local origin_size = origin:size()
    local origin_dim = origin:dim()

    -- Convert negative numbers to blue channel, positive numbers to red channel
    local color_r = torch.Tensor(origin_size):copy(origin):gt(0):float():cmul(origin)
    local color_g = torch.Tensor(origin_size):fill(0)
    local color_b = torch.Tensor(origin_size):copy(origin):lt(0):float():cmul(origin):mul(-1)
    -- print("red min and max:", torch.min(color_r), ";", torch.max(color_r))
    -- print("blue min and max:", torch.min(color_b), ";", torch.max(color_b))

    -- Construct the RGB tensor
    local color_tensor
    if origin_dim == 2 then
        color_tensor = torch.Tensor(3,origin_size[1],origin_size[2]):fill(0)
        color_tensor:select(1,1):copy(color_r)
        color_tensor:select(1,3):copy(color_b)
        -- print("origin_min:", torch.min(color_tensor:select(1,3)))
        -- print("origin_max:", torch.max(color_tensor:select(1,3)))
    elseif origin_dim == 3 then
        color_tensor = torch.Tensor(origin_size[1],3,origin_size[2],origin_size[3]):fill(0)
        color_tensor:select(2,1):copy(color_r)
        color_tensor:select(2,3):copy(color_b)
        -- print("origin_min:", torch.min(color_tensor:select(2,3)))
        -- print("origin_max:", torch.max(color_tensor:select(2,3)))
    end

    return color_tensor
end


--[[ Function
@description: Transform grey-scale map to 5 color heatmap
    Red: 1; Yellow: 0.5; Green: 0; Cyan:-0.5; Blue: -1
@input:
    origin: the original tensor to be visualized
@output:
    color_tensor: the colorized tensor transformed from the original tensor
@notes:
]]
function visu:GreyToHeatMap(origin)
    -- Normalize the original data
    origin = self:NormalizeTensor(origin)
    local origin_size = origin:size()
    local origin_dim = origin:dim()

    -- Get the RGB channels for 3 color heatmap
    -- Convert negative numbers to blue channel, positive numbers to red channel
    local color_r = torch.Tensor(origin_size):copy(origin):gt(0):float():cmul(origin)
    local color_b = torch.Tensor(origin_size):copy(origin):lt(0):float():cmul(origin):mul(-1)
    -- print("red min and max:", torch.min(color_r), ";", torch.max(color_r))
    -- print("blue min and max:", torch.min(color_b), ";", torch.max(color_b))
    -- Get the green channel for 3 color heatmap
    local color_g = torch.ones(origin_size):add(-1,torch.add(color_r,color_b))

    -- Construct the RGB tensor and convert to 5 color heatmap
    local color_tensor
    if origin_dim == 2 then
        color_tensor = torch.Tensor(3,origin_size[1],origin_size[2]):fill(0)
        color_tensor:select(1,1):copy(color_r):mul(2):clamp(0,1)
        color_tensor:select(1,2):copy(color_g):mul(2):clamp(0,1)
        color_tensor:select(1,3):copy(color_b):mul(2):clamp(0,1)
        -- print("origin_min:", torch.min(color_tensor:select(1,2)))
        -- print("origin_max:", torch.max(color_tensor:select(1,2)))
    elseif origin_dim == 3 then
        color_tensor = torch.Tensor(origin_size[1],3,origin_size[2],origin_size[3]):fill(0)
        color_tensor:select(2,1):copy(color_r):mul(2):clamp(0,1)
        color_tensor:select(2,2):copy(color_g):mul(2):clamp(0,1)
        color_tensor:select(2,3):copy(color_b):mul(2):clamp(0,1)
        -- print("origin_min:", torch.min(color_tensor:select(2,2)))
        -- print("origin_max:", torch.max(color_tensor:select(2,2)))
    end

    return color_tensor
end

--[[ Function
@description: concat a tensor with a dimensionality above 3 to a 3 dimensional tensor for visualization purpose
@input:
    data: the original tensor to be concated
@output:
    cat_data: the concated data
@notes:
    this version is just for concating a 4-dim tensor to 3-dim
]]
function visu:concat_3dim(data)
    local dim = data:dim()
    local size = data:size()
    -- local data_ = data:clone()
    local cat_data = data
    if dim > 3 then
        local n = size[1]
        cat_data = data[1]
        for i=2,n do
            cat_data = torch.cat(cat_data, data[i], 1)
        end
    end
    return cat_data
end

--[[ Function
@description: visualize an output tensor
@input:
    output_: the output tensor to be visualized
    simage_h_: the height of the visualized image
    simage_w_: the width of the visualized image
@output:
    imgDisplay: the visualized image tensor with normalized values for displaying
    output_: the dimension reduced original tensor
@notes:
]]
function visu:visualize_tensor(output_, simage_h_, simage_w_)
    -- display image arrays
    local output = output_:clone()

    if simage_h_ == 0 then
        output = self:concat_3dim(output)
        output_ = output:clone()

    -- transfer vectors into images to display
    else
        -- in case that the dimension is 0 at the beginning of a training
        if output:dim() > 1 then
            -- local o_size = output:size()
            -- output = output[o_size[1]]
            output = output[1]
            output_ = output:clone()
        end
        output = output:resize(simage_h_,simage_w_)
    end

    -- Get a normalized map
    if self.color_mode == "heatmap" then
        -- Conver to 5 color heatmap
        output = self:GreyToHeatMap(output)
    elseif self.color_mode == "three_color" then
        -- Convert to Red&Blue map
        output = self:GreyToRedBlueMap(output)
    else
        -- Normalize the grey-scale map
        output = self:NormalizeTensor(output)
        -- Transform to 0~1 from -1~1 for visualization
        output:add(1):div(2)
    end

    -- generate image array
    -- image.toDisplayTensor will normalize the data to between 0 and 1 with respect to each subimage by default.
    -- local imgDisplay = image.toDisplayTensor{input=output, padding=1, scaleeach=true, min=0, max=1, symmetric=false, saturate=false}
    -- Set min=0, max=1 to make the normalization ineffective, since the heatmap tensor has already been normalized.
    local imgDisplay = image.toDisplayTensor{input=output, padding=1, min=0, max=1}

    -- print("image_min:", torch.min(imgDisplay:select(1,2)))
    -- print("image_max:", torch.max(imgDisplay:select(1,2)))
    -- print("=============================================")

    return imgDisplay, output_
end


--[[ Function
@description: visualize the output of each layer in a network
@input:
    layer_num: the number of the layer to be visualized
@output:
    img_output: the visualized output tensor with normalized values for displaying
    original_output: the original output tensor
    img_sub: the visualized substraction tensor with normalized values for displaying
    original_sub: the original substraction tensor
@notes:
]]
function visu:visualize_single_layer_output(layer_num)
    -- local original_output = torch.Tensor(self.network:get(layer_num).output:float())
    local original_output = self.network:get(layer_num).output
    if original_output:dim() > 3 then
        original_output = original_output[1]
    end
    original_output = original_output:clone():float()
    -- For minibatch update, just visualize the case for the first sample
    -- if original_output:dim() > 3 then
    --     original_output = original_output[1]
    -- end
    -- print("layer", layer_num, "size:", #original_output)
    -- generate image array
    -- print(layer_num, "!!!!!!!!!outputs dim: ", #original_output)
    local img_output, original_output = self:visualize_tensor(original_output, self.simage_h[layer_num], self.simage_w[layer_num])

    local original_sub
    local img_sub
    if self.visu_output_sub then
        if self.pre_output and original_output:dim() == self.pre_output[layer_num]:dim() then
            -- print(original_output:size(), ";", self.pre_output[layer_num]:size())
            original_sub = original_output:clone():add(-1.0,self.pre_output[layer_num])
        else
            original_sub = original_output:clone():fill(0)
        end
        -- self.pre_output[layer_num] = original_output:clone()
        img_sub, original_sub = self:visualize_tensor(original_sub, self.simage_h[layer_num], self.simage_w[layer_num])
    end

    return img_output, original_output, img_sub, original_sub
end

-- TODO: Need more developments
--[[ Function
@description: visualize the weights of each layer in a network
@input:
    layer_num: the number of the layer to be visualized
@output:
    imgDisplay: the visualized image tensor with normalized values for displaying
    original_output: the original output tensor
@notes:
]]
function visu:visualize_single_layer_weights(layer_num)
    local weights_, grad_weights = self.network:get(self.w_layers[layer_num]):parameters()
    -- Both weights_ and grad_weights have two component tensors, one for weights, another for biases.

    -- local weights_, grad_weights = self.network:getParameters()
    -- [1]:weights; [2]:biase
    -- print("grad weights_ size:", #weights_)
    -- print("grad weights_ size:", #weights_[1])
    -- print("weights_ size:", weights_[1])
    -- print("weights_ size:", weights_[2])
    -- print("grad weights_ size:", #grad_weights[2])
    -- local weights = torch.Tensor(weights_[1]):float()
    local weights = weights_[1]:clone():float()
    -- print("layer", layer_num, "weight size:", weights:size())
    -- print("weights size:", weights:size())
    -- print("weights 1:", weights[1][1])
    -- local original_weights = weights:clone()
    -- print(layer_num, "!!!!!!!!!weights dim: ", #weights)

    -- generate image array
    local img_weights, original_weights = self:visualize_tensor(weights, 0, 0)

    local original_sub
    local img_sub
    if self.visu_weight_sub then
        if self.pre_weight and original_weights:dim() == self.pre_weight[layer_num]:dim() then
            -- print(original_output:size(), ";", self.pre_output[layer_num]:size())
            original_sub = original_weights:clone():add(-1.0,self.pre_weight[layer_num])
        else
            original_sub = original_weights:clone():fill(0)
        end
        -- self.pre_output[layer_num] = original_output:clone()
        img_sub, original_sub = self:visualize_tensor(original_sub, 0, 0)
    end

    return img_weights, original_weights, img_sub, original_sub
end


--[[ Function
@description: merge a source image to a destination image
@input:
    destin: the destination image
    sub_image: the source image
    verticle_mode: whether to merge images in the verticle direction, otherwise merge in the horizontal direction
@output:
    merged_image_: the merged image
@notes: the dimension of destin and sub_image should be consistent
]]
function visu:merge_image(destin, sub_image, verticle_mode)
    -- Use dim to adapt to different color modes
    local destin_dim = destin:dim()
    local sub_image_dim = sub_image:dim()

    local size_h = destin:size()[destin_dim-1]
    local size_w = destin:size()[destin_dim]
    local size_h0 = size_h
    local size_w0 = size_w
    local size_h_ = sub_image:size()[sub_image_dim-1]
    local size_w_ = sub_image:size()[sub_image_dim]

    -- merge images
    local merged_image_
    local size_h_add0
    local size_h_add1
    local size_w_add0
    local size_w_add1

    -- Set the region index for different merging modes
    if verticle_mode then
        if size_w_ > size_w then -- update the height of the merged image
            size_w = size_w_
        end
        size_h = size_h + size_h_ + 2 * self.padding_merged_image -- update the widthe of the merged image
        size_h_add0 = size_h0+self.padding_merged_image+1
        size_h_add1 = size_h0+self.padding_merged_image+size_h_
        size_w_add0 = 1
        size_w_add1 = size_w_

    else
        if size_h_ > size_h then -- update the height of the merged image
            size_h = size_h_
        end
        size_w = size_w + size_w_ + 2 * self.padding_merged_image -- update the widthe of the merged image
        size_h_add0 = 1
        size_h_add1 = size_h_
        size_w_add0 = size_w0+self.padding_merged_image+1
        size_w_add1 = size_w0+self.padding_merged_image+size_w_
    end

    -- Adapt to different color modes
    if destin_dim == 3 then
        merged_image_ = torch.Tensor(3,size_h,size_w):fill(1)
        merged_image_:sub(1, 3, 1, size_h0, 1, size_w0):copy(destin)
        merged_image_:sub(1, 3, size_h_add0, size_h_add1, size_w_add0, size_w_add1):copy(sub_image)
    else
        merged_image_ = torch.Tensor(size_h,size_w):fill(1)
        merged_image_:sub(1, size_h0, 1, size_w0):copy(destin)
        merged_image_:sub(size_h_add0, size_h_add1, size_w_add0, size_w_add1):copy(sub_image)
    end

    return merged_image_
end


--[[ Function
@description: add tail white area to make the width of the merged image divisible by 2
@input:
    origin_img: an image waiting to be wrapped
@output:
    wrapped_img: the wrapped image which is divisible by 2
@notes:
]]
function visu:wrap_img(origin_img)

    -- add tail white area to make the width of the merged image divisible by 2
    -- if the width is not divisible by 2, the ffmpeg cannot generate the video
    local add_width
    local add_height
    local tail_area_w
    local tail_area_h

    -- Adapt to different color modes
    if self.color_mode == "mono" then
        add_width = origin_img:size(2)%2
        add_height = origin_img:size(1)%2
        tail_area_w = torch.Tensor(self.padding_merged_image+add_width, self.padding_merged_image+add_width):fill(1)
        tail_area_h = torch.Tensor(self.padding_merged_image+add_height, self.padding_merged_image+add_height):fill(1)
    else
        add_width = origin_img:size(3)%2
        add_height = origin_img:size(2)%2
        tail_area_w = torch.Tensor(3, self.padding_merged_image+add_width, self.padding_merged_image+add_width):fill(1)
        tail_area_h = torch.Tensor(3, self.padding_merged_image+add_height, self.padding_merged_image+add_height):fill(1)
    end
    local wrapped_img = self:merge_image(origin_img, tail_area_w, false)
    wrapped_img = self:merge_image(wrapped_img, tail_area_h, true)

    -- save the merged image
    --image.save("entire_process_" .. step .. ".png", merged_image)

    return wrapped_img
end


--[[ Function
@description: visualize the output of each layer in a network
@input:
    step: the step sequence of the current process, for naming files
@output:
    merged_image: the merged image tensor with normalized values for displaying
    output_group: the original output group
@notes: the visualized images will be saved in the current working directory
]]
function visu:visualize_output_in_layers(step)

    -- definiation for merging images
    local size_h = self.padding_merged_image
    local size_w = self.padding_merged_image
    local merged_img_output
    local merged_img_sub

    -- Adapt to different color modes
    if self.color_mode == "mono" then
        merged_img_output = torch.Tensor(size_h,size_w):fill(1) -- normalized outputs for displaying
        if self.visu_output_sub then
            merged_img_sub = merged_img_output:clone()
        end
    else
        merged_img_output = torch.Tensor(3,size_h,size_w):fill(1) -- normalized outputs for displaying
        if self.visu_output_sub then
            merged_img_sub = merged_img_output:clone()
        end
    end

    local output_group = {} -- a table to save original output tensors
    local sub_group = {} -- a table to save original output substraction tensors
    local previous_output = {} -- a table for storing previous output tensors

    for i=1,self.n_modules do
        local img_output, original_output, img_sub, original_sub = self:visualize_single_layer_output(i)

        -- Save original data to the tables
        output_group[i] = original_output
        if original_sub then
            sub_group[i] = original_sub
        end
        --mattorch.save("output" .. i .. "_" .. step .. ".mat", original_output)

        -- merge images
        merged_img_output = self:merge_image(merged_img_output, img_output, false)
        if img_sub then
            merged_img_sub = self:merge_image(merged_img_sub, img_sub, false)
        end
    end

    -- update the variable saving the previous output frame
    if self.visu_output_sub then
        self.pre_interval_count = self.pre_interval_count + 1
        if self.pre_interval_count >= self.pre_interval then
            for i=1,self.n_modules do
                previous_output[i] = output_group[i]:clone()
            end
            self.pre_output = previous_output
            self.pre_interval_count = 0
        end
    end

    -- save the merged image
    --image.save("entire_process_" .. step .. ".png", merged_image)

    return merged_img_output, output_group, merged_img_sub, sub_group
end

-- TODO: Need more developments
--[[ Function
@description: visualize the weights of each layer in a network
@input:
    step: the step sequence of the current process, for naming files
@output:
    merged_image: the merged image tensor with normalized values for displaying
    output_group: the original output group
@notes: the visualized images will be saved in the current working directory
]]
function visu:visualize_weights_in_layers(step)
    -- definiation for merging images
    local size_h = self.padding_merged_image
    local size_w = self.padding_merged_image
    local merged_img_weight
    local merged_img_sub
    local m = #self.w_layers

    -- Adapt to different color modes
    if self.color_mode == "mono" then
        merged_img_weight = torch.Tensor(size_h,size_w):fill(1) -- normalized outputs for displaying
        if self.visu_weight_sub then
            merged_img_sub = merged_img_weight:clone()
        end
    else
        merged_img_weight = torch.Tensor(3,size_h,size_w):fill(1) -- normalized outputs for displaying
        if self.visu_weight_sub then
            merged_img_sub = merged_img_weight:clone()
        end
    end

    local weight_group = {} -- a table to save original output tensors
    local sub_group = {} -- a table to save original output substraction tensors
    local previous_weight = {} -- a table for storing previous output tensors

    for i=1,m do
        local img_weights, original_weights, img_sub, original_sub = self:visualize_single_layer_weights(i)

        -- Save original data to the tables
        weight_group[i] = original_weights
        if original_sub then
            sub_group[i] = original_sub
        end
        --mattorch.save("output" .. i .. "_" .. step .. ".mat", original_output)

        -- merge images
        merged_img_weight = self:merge_image(merged_img_weight, img_weights, false)
        if img_sub then
            merged_img_sub = self:merge_image(merged_img_sub, img_sub, false)
        end
    end

    -- update the variable saving the previous output frame
    if self.visu_weight_sub then
        self.pre_interval_count_w = self.pre_interval_count_w + 1
        if self.pre_interval_count_w >= self.pre_interval then
            for i=1,m do
                previous_weight[i] = weight_group[i]:clone()
            end
            self.pre_weight = previous_weight
            self.pre_interval_count_w = 0
        end
    end

    -- save the merged image
    --image.save("entire_process_" .. step .. ".png", merged_image)

    return merged_img_weight, weight_group, merged_img_sub, sub_group
end


--[[ Function
@description: substract two images and visualize after normalization
    image_y = abs(image_x0 - image_x1)
@input:
    image_x0: the first image tensor
    image_x1: the second image tensor
    step: the step sequence of the current process, for naming files
@output:
    merged_image: the merged substraction image tensor with normalized values for displaying
    output_group: the original substraction output group
@notes: the visualized substracted image will be saved in the current working directory
]]
function visu:substraction_image(image_x0, image_x1, step)
    local n = #image_x0
    local image_y = {}
    local merged_image

    -- Adapt to different color modes
    if self.color_mode == "mono" then
        merged_image = torch.Tensor(self.padding_merged_image,self.padding_merged_image):fill(1) -- normalized outputs for displaying
    else
        merged_image = torch.Tensor(3,self.padding_merged_image,self.padding_merged_image):fill(1) -- normalized outputs for displaying
    end

    for i=1,n do
        -- image_y[i] = (image_x0[i] - image_x1[i]):abs()
        image_y[i] = (image_x0[i] - image_x1[i])
        imgDisplay, image_y[i] = self:visualize_tensor(image_y[i], self.simage_h[i], self.simage_w[i])
        merged_image = self:merge_image(merged_image, imgDisplay, false)
    end
    --image.save("substraction_" .. step .. ".png", merged_image)
    --mattorch.save("substraction_" .. step .. ".mat", merged_image)
    return merged_image, image_y
end


--[[ Function
@description: visualize the outputs of a network
@input:
    step: the step sequence of the current process, for naming files
@output:
    merged_img_output: the merged image tensor with normalized values for displaying
    output_group: the original output group
    weight_group: the original weight group
    output_sub_group: the output substraction group
    weight_sub_group: the weight substraction group
@notes: the visualized images will be saved in the current working directory
]]
function visu:visualize_outputs(step)

    local output_image, output_sub_image
    local output_group, output_sub_group
    local merged_img_output

    -- Get the output image
    output_image, output_group, output_sub_image, output_sub_group = self:visualize_output_in_layers(step)
    if output_sub_image then
        merged_img_output = self:merge_image(output_image, output_sub_image, true)
    else
        merged_img_output = output_image
        output_sub_group = nil
    end

    -- Wrap image to make it divisible by 2
    merged_img_output = self:wrap_img(merged_img_output)

    return merged_img_output, output_group, output_sub_group
end


--[[ Function
@description: visualize the weights of a network
@input:
    step: the step sequence of the current process, for naming files
@output:
    merged_img_weight: the merged image tensor with normalized values for displaying
    output_group: the original output group
    weight_group: the original weight group
    output_sub_group: the output substraction group
    weight_sub_group: the weight substraction group
@notes: the visualized images will be saved in the current working directory
]]
function visu:visualize_weights(step)

    local weight_image, weight_sub_image
    local weight_group, weight_sub_group
    local merged_img_weight

    -- Get the weight image
    weight_image, weight_group, weight_sub_image, weight_sub_group = self:visualize_weights_in_layers(step)
    if weight_sub_image then
        merged_img_weight = self:merge_image(weight_image, weight_sub_image, true)
    else
        merged_img_weight = weight_image
        weight_sub_group = nil
    end

    -- Wrap image to make it divisible by 2
    merged_img_weight = self:wrap_img(merged_img_weight)

    return merged_img_weight, weight_group, weight_sub_group
end


--[[ Function
@description: visualize a network
@input:
    step: the step sequence of the current process, for naming files
@output:
    merged_image: the merged image tensor with normalized values for displaying
    output_group: the original output group
    weight_group: the original weight group
    output_sub_group: the output substraction group
    weight_sub_group: the weight substraction group
@notes: the visualized images will be saved in the current working directory
]]
function visu:visualize_network(step)

    -- Get the output image
    local merged_img_output, output_group, output_sub_group = self:visualize_outputs(step)

    -- Get the weight image
    local merged_img_weight, weight_group, weight_sub_group = self:visualize_weights(step)

    -- Merge all images
    local merged_image = self:merge_image(merged_img_output, merged_img_weight, true)

    -- Wrap image to make it divisible by 2
    merged_image = self:wrap_img(merged_image)

    return merged_image, output_group, weight_group, output_sub_group, weight_sub_group
end
