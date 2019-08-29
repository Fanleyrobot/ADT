--[[ File
@description:
    This class is for input pre-processing.
@version: V0.0
@author: Fangyi Zhang   email:gzzhangfangyi@gmail.com
@acknowledgement:
    ARC Centre of Excellence for Robotic Vision (ACRV)
    Queensland Univsersity of Technology (QUT)
@history:
    V0.00   03/06/2016  developed the first version
]]

require "image"
require "net/Scale"

local function create_network(args)
    -- Y (luminance)
    return nn.Scale(256, 256, true)
end

return create_network
