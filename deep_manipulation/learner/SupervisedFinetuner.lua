--[[ File
@description:
    This class is for end-to-end fine-tuning a combined network (p+c) using weighted losses.
@version: V0.15
@author: Fangyi Zhang   email:gzzhangfangyi@gmail.com
@acknowledgement:
    ARC Centre of Excellence for Robotic Vision (ACRV)
    Queensland Univsersity of Technology (QUT)
@history:
    V0.00   30/05/2016  developed the first version
    V0.01   06/06/2016  added cost histories
    V0.02   06/06/2016  updated the cost equation to (1/2m)*norm(2)
    V0.10   15/06/2016  added a function for testing
    V0.11   17/06/2016  made the code compatible to cases with/without cost histories when loading pre-trained models
    V0.12   19/06/2017  made the float() and cuda() compatible to both a tensor and a table of tensors.
    V0.13   21/06/2017  made the minibatchlearning compatible to both simple and complex networks
    V0.15   26/09/2018  re-organized
]]

if not dmn then
    require 'common/initenv'
end

local snl = torch.class('dmn.SupervisedFinetuner')


function snl:__init(args)
    self.feature_dim = args.feature_dim or {84,84}-- The feature dimention of the output from the simulator
    self.state_dim  = args.state_dim -- the dimensionality of the state represented as a vector.
    self.verbose    = args.verbose

    -- Validation
    self.validation_s = {}
    self.validation_l = {}
    self.validation_p = {}
    self.validation_extra_i = {}
    self.validation_extra_p = {}

    -- Learning rate annealing
    self.lr_start       = args.lr or 0.01 --Learning rate.
    self.lr             = self.lr_start
    self.lr_end         = args.lr_end or self.lr
    self.lr_endt        = args.lr_endt or 1000000
    self.wc             = args.wc or 0  -- L2 weight cost: lamda
    self.minibatch_size = args.minibatch_size or 1
    self.valid_size     = args.valid_size or 500

    --- Learning parameters
    self.fixed_layers   = args.fixed_layers or {11,13,15} -- Set the index of the weight-fixed layers
    self.m_fixed_layers = #self.fixed_layers
    self.n_replay       = args.n_replay or 1
    -- Number of steps after which learning starts.
    self.lr_startt    = args.lr_startt or 0
    self.n_vali_set     = args.n_vali_set or 5000
    self.hist_len       = args.hist_len or 1
    self.clip_delta     = args.clip_delta

    self.gpu            = args.gpu

    self.ncols          = args.ncols or 1  -- number of color channels in input
    self.input_dims     = args.input_dims or {self.hist_len*self.ncols, unpack(self.feature_dim)}
    -- self.input_dims     = args.input_dims or {self.hist_len*self.ncols, unpack(self.feature_dim)}
    self.preproc        = args.preproc  -- name of preprocessing network
    self.p_preproc      = args.p_preproc  -- name of preprocessing network

    self.network    = args.network or self:createNetwork()

    -- the cost histories
    self.cost_history = {}
    self.cost_sum_history = {}
    self.p_cost_history = {}
    self.p_cost_sum_history = {}

    -- check whether there is a network file
    local network_function
    if not (type(self.network) == 'string') then
        error("The type of the network provided in NeuralQLearner" ..
              " is not a string!")
    end

    local msg, err = pcall(require, self.network)
    if not msg then
        -- try to load saved agent
        local err_msg, exp = pcall(torch.load, self.network)
        if not err_msg then
            error("Could not find network file ")
        end
        self.network = exp.model
        -- load perception network
        self.p_network = exp.p_model
        -- if self.p_network == nil then
        --     Print("No perception network!!! Cannot fine-tune using weighted losses!!!")
        -- end
        if exp.cost_history then
            self.cost_history = exp.cost_history
        end
        if exp.cost_sum_history then
            self.cost_sum_history = exp.cost_sum_history
        end
        if exp.p_cost_history then
            self.p_cost_history = exp.p_cost_history
        end
        if exp.p_cost_sum_history then
            self.p_cost_sum_history = exp.p_cost_sum_history
        end


    else
        print('Creating Agent Network from ' .. self.network)
        self.network = err
        -- err stores the address of the function returned by the command of
        -- "require self.network",
        -- e.g., require 'convnet_atari3'
        self.network = self:network()
        -- run the function to create the network. Being a member function makes
        -- it load the self parameters as the args...

        -- The above two lines are equivalent to
        -- self.network = err(self)
    end

    if self.gpu and self.gpu >= 0 then
        self.network:cuda()
        self.p_network:cuda()
    else
        self.network:float()
        self.p_network:float()
    end

    -- Load preprocessing network.
    if not (type(self.preproc == 'string')) then
        error('The preprocessing is not a string')
    end
    msg, err = pcall(require, self.preproc)
    if not msg then
        error("Error loading preprocessing net")
    end
    self.preproc = err -- err is the address of the preprocessing function
    self.preproc = self:preproc()
    self.preproc:float()

    -- Load preprocessing network.
    if not (type(self.p_preproc == 'string')) then
        error('The preception preprocessing is not a string')
    end
    msg, err = pcall(require, self.p_preproc)
    if not msg then
        error("Error loading perception preprocessing net")
    end
    self.p_preproc = err -- err is the address of the preprocessing function
    self.p_preproc = self:p_preproc()
    self.p_preproc:float()

    if self.gpu and self.gpu >= 0 then -- why two times????? this has been run in previous procedures
        self.network:cuda()
        self.p_network:cuda()
        self.tensor_type = torch.CudaTensor
    else
        self.network:float()
        self.p_network:float()
        self.tensor_type = torch.FloatTensor
    end

    self.numSteps = 0 -- Number of perceived states.

    self.w, self.dw = self.network:getParameters()
    self.dw:zero()

    self.deltas = self.dw:clone():fill(0)

    self.tmp= self.dw:clone():fill(0)
    self.g  = self.dw:clone():fill(0)
    self.g2 = self.dw:clone():fill(0)

    -- perception network weight relevant variables
    self.p_w, self.p_dw = self.p_network:getParameters()
    self.shared_w_dim = self.p_w:size(1)
    self.partial_p = false
    -- self.shared_w_dim = 105728
    -- self.partial_p = true
    -- print("shared weights dim:", self.shared_w_dim)
    -- print("CNN sub weights:", self.p_w:narrow(1, 44451, 6))
    -- print("E2E sub weights:", self.w:narrow(1, 44451, 6))
    self.p_dw:zero()

    self.p_deltas = self.p_dw:clone():fill(0)

    self.p_tmp= self.p_dw:clone():fill(0)
    self.p_g  = self.p_dw:clone():fill(0)
    self.p_g2 = self.p_dw:clone():fill(0)
    self.p_scalar = args.p_weight or 0.8
    self.c_scalar = 1 - self.p_scalar
    print("Perception weight: ", self.p_scalar)
    print("Control/E2E weight: ", self.c_scalar)
end


function snl:reset(state)
    if not state then
        return
    end
    self.network = state.model
    self.w, self.dw = self.network:getParameters()
    self.dw:zero()
    self.numSteps = 0
    print("RESET STATE SUCCESFULLY")
end


function snl:preprocess(rawstate)
    -- print(rawstate:size())
    if self.preproc then
        return self.preproc:forward(self:float(rawstate))
        -- return rawstate:float():clone():reshape(self.state_dim) -- for fc network
    end

    return rawstate
end


function snl:p_preprocess(rawstate)
    -- print(rawstate:size())
    if self.p_preproc then
        return self.p_preproc:forward(self:float(rawstate))
        -- return rawstate:float():clone():reshape(self.state_dim) -- for fc network
    end

    return rawstate
end


function snl:getDelta(args)
    local s, l, delta
    local p

    s = args.s -- image
    l = args.l -- label
    local network = args.network or self.network

    -- Compute predictions
    p = network:forward(s):float()
    -- print("p: ",p:size())

    -- Compute error = l - p
    delta = l:clone():float()
    -- print("delta: ",delta:size())
    delta:add(-1, p)

    -- Clip the delta, avoiding too big or small deltas
    if self.clip_delta then
        delta[delta:ge(self.clip_delta)] = self.clip_delta
        delta[delta:le(-self.clip_delta)] = -self.clip_delta
    end

    if self.gpu >= 0 then delta = delta:cuda() end

    return delta
end


function snl:supLearnMinibatch(s, l, p, extra_i, extra_p)
    -- Perform a minibatch Q-learning update:
    -- w += alpha * (r + gamma max Q(s2,a2) - Q(s,a)) * dQ(s,a)/dw
    if type(s) == 'table' then
        assert(s[1]:size(1) == self.minibatch_size)
        assert(s[1]:size(1) == l:size(1))
    else
        assert(s:size(1) == self.minibatch_size)
        assert(s:size(1) == l:size(1))
    end

    if self.gpu >= 0 then s = self:cuda(s) end

    local delta = self:getDelta{s=s, l=l}

    -- zero gradients of parameters
    self.dw:zero()

    -- get new gradient
    -- the function of backward will compute the gradient of each weight according to the deltas
    -- deltas = y - h_theta(x)
    -- this means the minus sign was integrated in to the deltas
    -- therefore, w += ... not w -= ...
    -- compute dw, here we set m=1, which should be the minibatch size (32) according to the introduced definition, but it is not a hard condition.
    self.network:backward(s, delta) -- backward(input, gradOutput, [m]), m is 1 by default.

    -- Get the gradients for the perception cost function
    if p ~= nil then
        -- print("Perception Cost")
        local p_s = s[2]
        local p_l = p
        if extra_i ~= nil then
        --   print("Extra Images")
        --   print(#extra_i)
          extra_i = self:p_preprocess(extra_i):float()
          if self.gpu >= 0 then
              extra_i = extra_i:cuda()
            --   extra_p = extra_p:cuda()
          end
          -- p_s = torch.cat(extra_i,p_s,1)
          -- p_l = torch.cat(extra_p,p_l,1)
          p_s = extra_i
          p_l = extra_p
        end
        delta = self:getDelta{s=p_s, l=p_l, network=self.p_network}

        -- zero gradients of parameters
        self.p_dw:zero()

        -- get new gradient
        -- the function of backward will compute the gradient of each weight according to the deltas
        -- deltas = y - h_theta(x)
        -- this means the minus sign was integrated in to the deltas
        -- therefore, w += ... not w -= ...
        -- compute dw, here we set m=1, which should be the minibatch size (32) according to the introduced definition, but it is not a hard condition.
        self.p_network:backward(p_s, delta) -- backward(input, gradOutput, [m]), m is 1 by default.

        -- Compose the new gradients
        -- 0.2*self.dw+0.8*self.p_dw

        local dw_share = self.dw:narrow(1,1,self.shared_w_dim)
        dw_share:mul(self.c_scalar):add(self.p_scalar, self.p_dw:narrow(1,1,self.shared_w_dim))
    end

    -- add weight cost to gradient (L2 weight decay)
    -- Since the minus sign has been integrated into the dw,
    -- accordingly, we also have to integrate the minus sign into the weight cost, i.e., -self.wc
    -- which is normally +self.wc=lambda/m, i.e., (lambda/m)*w,
    -- here we set m=1, which should be the minibatch size (32) according to the introduced definition, but it is not a hard condition.
    self.dw:add(-self.wc, self.w)

    -- compute linearly annealed learning rate
    local t = math.max(0, self.numSteps - self.lr_startt)
    self.lr = (self.lr_start - self.lr_end) * (self.lr_endt - t)/self.lr_endt +
                self.lr_end
    self.lr = math.max(self.lr, self.lr_end)
    -- Intergrate m (the number of samples in the minibatch) into the learning rate
    -- making the dw to averaging dw, instead of their sum
    self.lr = self.lr / self.minibatch_size

    -- use gradients
    -- a of variant of RMSprop, an adaptive learning rate method
    -- compute the term of sqrt(g2 - g^2 +0.01)
    -- g = 0.95*g + 0.05*dw
    -- g2 = 0.95*g2 + 0.05*dw^2
    self.g:mul(0.95):add(0.05, self.dw)
    self.tmp:cmul(self.dw, self.dw)
    self.g2:mul(0.95):add(0.05, self.tmp)
    self.tmp:cmul(self.g, self.g)
    self.tmp:mul(-1)
    self.tmp:add(self.g2)
    self.tmp:add(0.01)
    self.tmp:sqrt()

    -- Set the dw of the fixed layers to zeros
    for i=1,self.m_fixed_layers do
        -- self.network:get(self.fixed_layers[i]).gradWeight:zero()
        -- self.network:get(self.fixed_layers[i]).gradBias:zero()
        self.network:get(self.fixed_layers[i]):zeroGradParameters()
    end
    -- print("sample weight!!!!!!!!: ", self.network:get(15).weight[1][1])
    -- print("sample weight!!!!!!!!: ", self.network:get(2).weight[1][1][1][1])

    -- accumulate update
    -- w += lr * dw ./ sqrt(g2 - g^2 +0.01)
    self.deltas:mul(0):addcdiv(self.lr, self.dw, self.tmp) -- 0 + lr * dw ./ tmp
    self.w:add(self.deltas)

    -- Update the weights in the perception network accordingly
    if p ~= nil then
        if self.partial_p == true then
          self.p_dw:add(-self.wc, self.p_w)

          -- use gradients
          -- a of variant of RMSprop, an adaptive learning rate method
          -- compute the term of sqrt(g2 - g^2 +0.01)
          -- g = 0.95*g + 0.05*dw
          -- g2 = 0.95*g2 + 0.05*dw^2
          self.p_g:mul(0.95):add(0.05, self.p_dw)
          self.p_tmp:cmul(self.p_dw, self.p_dw)
          self.p_g2:mul(0.95):add(0.05, self.p_tmp)
          self.p_tmp:cmul(self.p_g, self.p_g)
          self.p_tmp:mul(-1)
          self.p_tmp:add(self.p_g2)
          self.p_tmp:add(0.01)
          self.p_tmp:sqrt()

          -- accumulate update
          -- w += lr * dw ./ sqrt(g2 - g^2 +0.01)
          self.p_deltas:mul(0):addcdiv(self.lr, self.p_dw, self.p_tmp) -- 0 + lr * dw ./ tmp
          self.p_w:add(self.p_deltas)
        end

        local p_w_share = self.p_w:narrow(1,1,self.shared_w_dim)
        p_w_share:copy(self.w:narrow(1,1,self.shared_w_dim))
    end

end


function snl:sample_validation_data(s, l, p, extra_i, extra_p)
    local s_ = self:float(self:preprocess(s))
    self.validation_s = s_
    self.validation_l = l
    self.validation_p = p

    if self.gpu >= 0 then
        self.validation_s = self:cuda(self.validation_s)
    end

    if extra_i then
        -- print("Collected extra dataset")
        -- print(type(extra_i))
        extra_i = self:float(self:p_preprocess(extra_i))
        -- print(type(extra_i))
        self.validation_extra_i = extra_i
        self.validation_extra_p = extra_p
        if self.gpu >= 0 then
            self.validation_extra_i = self:cuda(self.validation_extra_i)
        end
    end
end

-- Change Tensor or Tensors in a table into float type.
function snl:float(s)
  if type(s) == 'table' then
    local s_ = {}
    for k,v in pairs(s) do
      s_[k] = v:float()
    end
    return s_
  else
    return s:float()
  end
end

-- Change Tensor or Tensors in a table into cuda type.
function snl:cuda(s)
  if type(s) == 'table' then
    local s_ = {}
    for k,v in pairs(s) do
      s_[k] = v:cuda()
    end
    return s_
  else
    return s:cuda()
  end
end


function snl:compute_validation_statistics(s, l, p, extra_i, extra_p)

    if not s then
        s = self.validation_s
        l = self.validation_l
        p = self.validation_p
        extra_i = self.validation_extra_i
        extra_p = self.validation_extra_p
    end

    local delta = self:getDelta{s=s, l=l}

    delta = delta:float()
    delta:pow(2)
    local a = 1 / (2 * self.n_vali_set) -- 1/2m
    local cost = torch.sum(delta,1):mul(a)
    local cost_sum = torch.sum(cost)

    local p_s = s[2]
    local p_l = p
    if extra_i ~= nil and type(extra_i) ~= 'table' then
      p_s = torch.cat(extra_i,p_s,1)
      p_l = torch.cat(extra_p,p_l,1)
    end
    delta = self:getDelta{s=p_s, l=p_l, network=self.p_network}

    delta = delta:float()
    delta:pow(2)
    local p_cost = torch.sum(delta,1):mul(a)
    local p_cost_sum = torch.sum(p_cost)

    -- Update the cost histories
    local index = #self.cost_sum_history + 1
    self.cost_history[index] = cost
    self.cost_sum_history[index] = cost_sum
    self.p_cost_history[index] = p_cost
    self.p_cost_sum_history[index] = p_cost_sum

    return cost, cost_sum, p_cost, p_cost_sum
end


function snl:train(s, l, p, extra_i, extra_p)
    -- Preprocess state (will be set to nil if terminal)
    -- print(#s)
    -- s = torch.Tensor(s)
    local s_ = self:float(self:preprocess(s))

    -- Do the learning
    for i = 1, self.n_replay do
        -- self:supLearnMinibatch(s_, l)
        self:supLearnMinibatch(s_, l, p, extra_i, extra_p)
    end

    self.numSteps = self.numSteps + 1
end


function snl:test(s)
    -- Preprocess state (will be set to nil if terminal)
    -- print(#s)
    -- s = torch.Tensor(s)
    local s_ = self:float(self:preprocess(s))

    if self.gpu >= 0 then s_ = self:cuda(s_) end

    -- Compute predictions
    local p = self.network:forward(s_):float()

    return p

end


function snl:createNetwork()
    local n_hid = 128
    local mlp = nn.Sequential()
    mlp:add(nn.Reshape(self.hist_len*self.ncols*self.state_dim))
    mlp:add(nn.Linear(self.hist_len*self.ncols*self.state_dim, n_hid))
    mlp:add(nn.Rectifier())
    mlp:add(nn.Linear(n_hid, n_hid))
    mlp:add(nn.Rectifier())
    mlp:add(nn.Linear(n_hid, self.n_actions))

    return mlp
end


function snl:_loadNet()
    local net = self.network
    if self.gpu then
        net:cuda()
    else
        net:float()
    end
    return net
end


function snl:init(arg)
    self.network = self:_loadNet()
    -- Generate targets.
end


function snl:report()
    print(get_weight_norms(self.network))
    print(get_grad_norms(self.network))
end
