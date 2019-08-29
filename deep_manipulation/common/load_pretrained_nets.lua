--[[ File
@description:
    This file is for loading pre-trained networks.
@version: V0.00
@author: Fangyi Zhang   email:gzzhangfangyi@gmail.com
@acknowledgement:
    ARC Centre of Excellence for Robotic Vision (ACRV)
    Queensland Univsersity of Technology (QUT)
@history:
    V0.00   17/05/2016  developed the first version
]]

-- require 'loadcaffe'


-- A function to load a pretrained network file
function load_pretrained_network(network, index)
    if not index then
      index = 'model'
    end
    -- load a pre-trained control network
    local msg, err = pcall(require, network)
    local best_model
    local model
    if not msg then
        -- try to load saved agent
        local err_msg, exp = pcall(torch.load, network)
        if not err_msg then
            error("Could not find network file ")
        end
        -- best_model = exp.best_model
        model = exp[index]
    else
        print("Wrong network name")
    end

    return model

end