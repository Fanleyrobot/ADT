--[[ File
@description:
    This class is for image scaling.
@version: V0.0
@author: Fangyi Zhang   email:gzzhangfangyi@gmail.com
@acknowledgement:
    ARC Centre of Excellence for Robotic Vision (ACRV)
    Queensland Univsersity of Technology (QUT)
@history:
    V0.00   03/06/2016  developed the first version
]]

require "nn"
require "image"
require "torch"

local scale = torch.class('nn.Scale', 'nn.Module')


function scale:__init(height, width)
    self.height = height
    self.width = width
end

-- Optimized for minibatch scaling
function scale:forward(x)
    local n_sample, x_backup, x_scaled

    if x:dim() > 3 then
        n_sample = x:size(1)
        x_backup = x:clone()
        -- x_scaled = torch.Tensor(n_sample, 1, self.height, self.width)
        x_scaled = torch.Tensor(n_sample, 3, self.height, self.width)
    end

    if n_sample then -- if x is a batch
        for i=1,n_sample do
            x = x_backup[i]
            -- if x:size(1) == 3 then
            --     x = image.rgb2y(x)
            -- end
            --image.save('scaled1.png',x) -- Added by Fangyi Zhang
            x = image.scale(x, self.width, self.height, 'bilinear')
            x_scaled[i] = x
            --image.save('scaled2.png',x) -- Added by Fangyi Zhang
            -- print("x:", x:size())
        end
        x = x_scaled

    else -- if x is a single frame
        -- if x:size(1) == 3 then
        --     x = image.rgb2y(x)
        -- end
        --image.save('scaled1.png',x) -- Added by Fangyi Zhang
        x = image.scale(x, self.width, self.height, 'bilinear')
    end

    -- print("x:", x:size())
    return x
end

function scale:updateOutput(input)
    return self:forward(input)
end

function scale:float()
end
