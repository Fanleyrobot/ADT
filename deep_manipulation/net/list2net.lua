--[[ File
@description:
    This file is for constructing a conv net in accordance to list modules.
@version: V0.0
@author: Fangyi Zhang   email:gzzhangfangyi@gmail.com
@acknowledgement:
    ARC Centre of Excellence for Robotic Vision (ACRV)
    Queensland Univsersity of Technology (QUT)
@history:
    V0.00   05/07/2017  developed the first version
]]

require "common/initenv"

function create_network(args)
  local net = nn.Sequential()
  -- net:add(nn.Reshape(unpack(args.input_dims)))
  for i,item in ipairs(args.list_modules) do
    item[2].name = item[1]
    net:add(item[2])
  end

  if args.gpu >=0 then
      net:cuda()
  end
  if args.verbose >= 2 then
      print(net)
  end

  return net
end
