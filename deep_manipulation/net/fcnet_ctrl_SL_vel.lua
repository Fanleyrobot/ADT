--[[ File
@description:
    This function is for creating a FC network.
@version: V0.00
@author: Fangyi Zhang   email:gzzhangfangyi@gmail.com
@acknowledgement:
    ARC Centre of Excellence for Robotic Vision (ACRV)
    Queensland Univsersity of Technology (QUT)
@history:
    V0.00   30/05/2017  developed the first version
]]

require 'net/fcnet'

return function(args)
    -- fc network for 3 DoF Manipulation

    args.n_hid          = {400,300}
    args.nl             = nn.Rectifier

    -- Temporary ways of making it compatible to both Supervised Learner and NeuralQLearner.
    -- In Neural Q Learner, the n_actions is the output dim.
    -- TODO: Re-organize
    if not args.n_actions then
      args.n_actions      = 7
    end

    return create_network(args)
end
