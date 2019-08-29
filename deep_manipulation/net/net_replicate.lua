--[[ File
@description:
    This function is for creating a replication preprocessing network.
@version: V0.00
@author: Fangyi Zhang   email:gzzhangfangyi@gmail.com
@acknowledgement:
    ARC Centre of Excellence for Robotic Vision (ACRV)
    Queensland Univsersity of Technology (QUT)
@history:
    V0.00   04/01/2016  developed the first version

]]

require "nn"

local function create_network(args)
    return nn.Replicate(1)
end

return create_network
