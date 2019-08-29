--[[ File
@description:
    This class is for input pre-processing for a complex network.
@version: V0.0
@author: Fangyi Zhang   email:gzzhangfangyi@gmail.com
@acknowledgement:
    ARC Centre of Excellence for Robotic Vision (ACRV)
    Queensland Univsersity of Technology (QUT)
@history:
    V0.00   01/06/2017  developed the first version
]]

require "image"
require "net/Scale"

local function create_network(args)
  local preproc_net = nn.ParallelTable()
  preproc_net:add(nn.Copy())
  preproc_net:add(nn.Scale(256, 256, true))
  return preproc_net
end

return create_network
