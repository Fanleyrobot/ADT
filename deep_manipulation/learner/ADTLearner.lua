--[[ File
@description:
    This is a general class for ADT.
    This class uses the DistKLDivCriterion in training.
    This class uses a smarter way to coordinate G and D with an exponantial function based PID controller.
    This class is compatible to networks with or without shortcuts.
@version: V0.42
@author: Fangyi Zhang   email:gzzhangfangyi@gmail.com
@acknowledgement:
    ARC Centre of Excellence for Robotic Vision (ACRV)
    Queensland Univsersity of Technology (QUT)
@history:
    V0.00   10/01/2018  developed the first version
    V0.01   15/01/2018  fixed the bug of unexpected delta clips when calculating statistic results
    V0.02   16/01/2018  made the codes flexible for networks with and without shortcuts
    V0.03   17/01/2018  changed the average n to max n which will help stablize the GANs
    V0.04   17/01/2018  made the codes flexible for both unsupervised and semi-supervised adaptation
    V0.05   22/01/2018  added weight normalization for more accurately weighting the perception and domain imiting losses
    V0.10   03/05/2018  changed trainE2E_semi to put samples for end-to-end training into the batch for ad loss
    V0.20   01/08/2018  merged semi-supervised adapter and finetuner into one class: ADTLearner
    V0.25   22/05/2019  fixed the bug that the agent mode cannot be set via string arguments
    V0.30   23/05/2019  added the adversarial_mode setting to enable both ADDA and DC approaches
    V0.35   24/05/2019  added a printingSettings function to display all settings in one function
    V0.40   29/05/2019  fixed the bug that the weights in encoders were not shared
    V0.42   11/06/2019  added PID_map_scalar to set the strength of the adversarial cmd
]]

if not dmn then
    require 'common/initenv'
end

local adt = torch.class('dmn.ADTLearner')


function adt:__init(args)
    -- State dimention
    self.feature_dim = args.feature_dim or {84,84}-- The feature dimention of the output from the simulator
    self.state_dim  = args.state_dim -- the dimensionality of the state represented as a vector.
    self.verbose    = args.verbose

    -- Declare Variables for validation sets
    -- self.validation_s_sim -- simulated images for validation (perception)
    -- self.validation_s_real -- real images for validation (perception)
    -- self.validation_pose_sim -- ground-truth target object poses for simulated images (perception)
    -- self.validation_pose_real -- ground-truth target object poses for real images (perception)
    -- self.validation_s -- images for validation (end-to-end)
    -- self.validation_l -- ground-truth velocity labels for images (end-to-end)

    -- Learning rate annealing
    self.lr_start       = args.lr or 0.01 --learning rate.
    self.lr             = self.lr_start
    self.lr_end         = args.lr_end or self.lr
    self.lr_endt        = args.lr_endt or 1000000
    self.wc             = args.wc or 0  -- L2 weight cost: lamda
    self.minibatch_size = args.minibatch_size or 1
    self.valid_size     = args.valid_size or 500 -- validation set size

    -- Learning parameters
    self.fixed_layers   = args.fixed_layers or {} -- Set the index of the weight-fixed layers
    self.m_fixed_layers = #self.fixed_layers
    self.n_replay       = args.n_replay or 1
    -- Number of steps after which learning starts.
    self.lr_startt    = args.lr_startt or 0
    self.n_vali_set     = args.n_vali_set or 5000
    self.hist_len       = args.hist_len or 1
    self.clip_delta     = args.clip_delta

    self.gpu            = args.gpu -- GPU index to use: -1: cpu; 0~:gpu

    self.ncols          = args.ncols or 1  -- number of color channels in input
    self.input_dims     = args.input_dims or {self.hist_len*self.ncols, unpack(self.feature_dim)}
    -- self.input_dims     = args.input_dims or {self.hist_len*self.ncols, unpack(self.feature_dim)}
    self.preproc        = args.preproc  -- name of preprocessing network
    self.network        = args.network or self:createNetwork()

    -- Variables to store cost histories
    self.hist_Lp_vector_real = {} -- History of the perception loss vector, real domain
    self.hist_Lp_sum_real = {} -- History of the sum of the elements in the perception loss vector, real domain
    self.hist_Lp_vector_sim = {} -- History of the perception loss vector, simulation domain
    self.hist_Lp_sum_sim = {} -- History of the sum of the elements in the perception loss vector, simulation domain
    self.hist_LD = {} -- History of the discriminative loss
    self.hist_Le2e_vector_sim = {} -- History of the end-to-end loss vector, simulation domain
    self.hist_Le2e_sum_sim = {} -- History of the sum of the elements in the end-to-end loss vector, simulation domain

    -- Training or testing mode
    self.mode = args.mode or 1 -- three mode options: 1:e2e, 2:perception, 3:ctrl

    -- Settings for network types and learning modes
    self.shortcut_mode = args.shortcut_mode or false
    self.semi_supervised = args.semi_supervised or false
    self.adversarial_mode = args.adversarial_mode or 1 -- what adversarial loss to use, 1:ADDA, 2:DC, to add more
    self.parallel_mode = args.parallel_mode or false -- whether to update source encoder in the mean time

    -- Parameters for the coordination of Discriminater, Encoders and Perception losses
    self.d_episode_steps = args.d_episode_steps or 1
    self.e_episode_steps = args.e_episode_steps or 1
    self.encoder_start = args.encoder_start or 0

    self.d_lr_discount = args.d_lr_discount or 0.5
    self.use_PID = args.use_PID or false -- whether to use the PID controller
    self.ad_weight = 1.0  -- current weight for the adversarial discriminative or domain confusion loss, it is 1.0 by default
    -- self.task_weight = 1 - self.ad_weight -- current weight for the task loss, i.e., pose regression loss
    self.task_weight = 1.0 -- the weight for the task loss, it is 1.0 by default

    -- Initialize the variables for a PID controller to coordinate D and G for ADDA
    self.e_d = 0  -- the error between goal loss and current loss
    self.integral_e_d = 0  -- the integral of errors
    self.derivative_e_d = 0  -- the derivative of errors
    self.e_d_previous = 0 -- the previous error
    self.Kp = args.Kp or 0.4
    self.Ki = args.Ki or 0.008
    self.Kd = args.Kd or 0
    self.desired_d_loss = args.desired_d_loss or 0.28 -- the desired discriminative loss for the PID controller
    self.PID_map_scalar = args.PID_map_scalar or 0.02 -- a scalar weight to control the mapping from u to gama, i.e., strength of the adversarial cmd

    -- Initialize gama parameters for weighted losses for e2e fine-tuning
    if self.mode == 1 then -- e2e mode
      self.p_scalar = args.p_weight or 0.9
      self.c_scalar = 1 - self.p_scalar
    end

    -- self.d_loss_thr = args.d_loss_thr or 0.1
    -- print("D Loss Threshold: ", self.d_loss_thr)
    self.d_loss_filter_length = args.d_loss_filter_length or 3
    -- self.n_accum_smaller_d_lss = 0
    self.latest_n_d_loss = {}

    self.numSteps = 0 -- Number of perceived states.

    -- Encoders
    self.encoder_tg = self:EmbodyNetwork{network=self.network, index='encoder_tg', to_optimize=true}
    -- self.encoder_sr = self:EmbodyNetwork{network=self.network, index='encoder_sr', to_optimize=false}
    self.encoder_sr = self:EmbodyNetwork{network=self.network, index='encoder_sr', to_optimize=true}
    -- self.encoder_sr = self:EmbodyNetwork(self.network, 'model_sr')

    -- Pose Regressor
    self.pose_net = self:EmbodyNetwork{network=self.network, index='pose_net', to_optimize=true}

    -- Discriminater
    self.discriminater = self:EmbodyNetwork{network=self.network, index='d_net', to_optimize=true}

    -- Contrller
    if self.mode ~= 2 then -- not perception mode
      self.ctrl_net = self:EmbodyNetwork{network=self.network, index='ctrl_net', to_optimize=true}
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

    if self.gpu and self.gpu >= 0 then
        self.tensor_type = torch.CudaTensor
    else
        self.tensor_type = torch.FloatTensor
    end

    -- Set the networks to evaluation mode in case the networks contain dropout or batch normalization layers
    self.encoder_tg.network:evaluate()
    self.encoder_sr.network:evaluate()
    self.pose_net.network:evaluate()
    self.discriminater.network:evaluate()
    if self.ctrl_net then -- TODO: Add a function to test whether a network exists
      self.ctrl_net.network:evaluate()
    end

    -- Initialize an cross entropy entity for loss calculation
    -- self.criterion_CrossEntropy = nn.CrossEntropyCriterion()
    -- print("criterion batch mode: ",self.criterion_CrossEntropy.nll.sizeAverage)
    self.criterion_KL = nn.DistKLDivCriterion() -- KL divergence loss
    self.criterion_MSE = nn.MSECriterion() -- Mean squared loss

    self:printSettings() -- Print out all settings
end

-- Print settings
function adt:printSettings()
  print("~~~~~~ Agent Mode Settings ~~~~~~")
  print("Agent Mode: ", self.mode)
  if self.mode == 1 then -- e2e mode
    print("Perception weight: ", self.p_scalar)
    print("Control/E2E weight: ", self.c_scalar)
  end
  print("Shortcut Mode: ", self.shortcut_mode)
  print("Semi-supervised Mode: ", self.semi_supervised)
  print("Adversarial Mode: ", self.adversarial_mode)
  print("Parallel Mode: ", self.parallel_mode)

  print("~~~~~ PI Controller Settings ~~~~~")
  print("Whether to use the PID controller: ", self.use_PID)
  print("Desired D Loss: ", self.desired_d_loss)
  print("Kp: ", self.Kp)
  print("Ki: ", self.Ki)
  print("Kd: ", self.Kd)
  print("PID Mapping Scalar: ", self.PID_map_scalar)

  print("~~~~~~~~~ Other Settings ~~~~~~~~~")
  print("Discriminater learning rate discount: ", self.d_lr_discount)
  print("No. of D Losses for smart weights: ", self.d_loss_filter_length)
  print("Default task weight: ", self.task_weight)
  print("Default Ad weight: ", self.ad_weight)
  print("d_episode_steps: ", self.d_episode_steps)
  print("e_episode_steps: ", self.e_episode_steps)
  print("encoder start: ", self.encoder_start)
  print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

end

-- Embody a network according to the settings in args
function adt:EmbodyNetwork(args)
  local network = args.network
  local index = args.index or 'model'
  local to_optimize = args.to_optimize or false

  -- check whether there is a network file
  if not (type(network) == 'string') then
      error("The type of the network provided is not a string!")
  end

  local agent = {}
  local msg, err = pcall(require, network)
  if not msg then
      -- try to load saved agent
      local err_msg, exp = pcall(torch.load, network)
      if not err_msg then
          error("Could not find network file ")
      end
      agent.network = exp[index]

  else
      print('Creating Agent Network from ' .. network)
      -- err stores the address of the function returned by the command of
      agent.network = err(self)
      -- run the function to create the network. Being a member function makes
      -- it load the self parameters as the args...
  end

  if self.gpu and self.gpu >= 0 then
      agent.network:cuda()
  else
      agent.network:float()
  end

  if to_optimize then
    -- Initialize the variables for adam optimizer
    agent.w, agent.dw = agent.network:getParameters()
    agent.dw:zero()

    agent.deltas = agent.dw:clone():fill(0)

    agent.tmp= agent.dw:clone():fill(0)
    agent.g  = agent.dw:clone():fill(0)
    agent.g2 = agent.dw:clone():fill(0)
  end

  return agent
end

-- A function for preprocessing raw inputs, particularly for the image components
function adt:preprocess(rawstate)
    -- print(rawstate:size(1))
    if self.preproc then
      -- return self.preproc:forward(self:float(rawstate))
      if type(rawstate) == 'table' then
        local s = self.preproc:forward(self:float(rawstate[2])):clone()
        return {rawstate[1],s}
      else
        return self.preproc:forward(self:float(rawstate)):clone()
        -- return rawstate:float():clone():reshape(self.state_dim) -- for fc network
      end
    end

    return rawstate
end

-- Get delta based a criterion with possible clips
-- deltas = predictions - labels
function adt:getDelta(p, l, criterion, nonclip)
  local delta = criterion:backward(p, l)

  -- Clip the delta, avoiding too big or small deltas
  if self.clip_delta and not nonclip then
      delta[delta:ge(self.clip_delta)] = self.clip_delta
      delta[delta:le(-self.clip_delta)] = -self.clip_delta
  end

  if self.gpu >= 0 then delta = delta:cuda() end

  return delta
end

-- Get Losses based a criterion
function adt:getLoss(p, l, criterion)

  local loss = criterion:forward(p, l)

  return loss
end

-- Get gradients for a network
function adt:getGradient(input, delta, network)
  network.dw:zero()

  local gradInput= network.network:backward(input, delta) -- backward(input, gradOutput, [m]), m is 1 by default.

  return network.dw, gradInput

end

-- An Ramprop variant for optimization
function adt:AdamOptimizer(network, lr)

  -- add weight cost to gradient (L2 weight decay)
  -- Since the minus sign has been integrated into the dw,
  -- accordingly, we also have to integrate the minus sign into the weight cost, i.e., -self.wc
  -- which is normally +self.wc=lambda/m, i.e., (lambda/m)*w,
  -- here we set m=1, which should be the minibatch size (32) according to the introduced definition, but it is not a hard condition.
  network.dw:add(self.wc, network.w)

  -- use gradients
  -- a of variant of RMSprop, an adaptive learning rate method
  -- compute the term of sqrt(g2 - g^2 +0.01)
  -- g = 0.95*g + 0.05*dw
  -- g2 = 0.95*g2 + 0.05*dw^2
  network.g:mul(0.95):add(0.05, network.dw)
  network.tmp:cmul(network.dw, network.dw)
  network.g2:mul(0.95):add(0.05, network.tmp)
  network.tmp:cmul(network.g, network.g)
  network.tmp:mul(-1)
  network.tmp:add(network.g2)
  network.tmp:add(0.01)
  network.tmp:sqrt()

  -- -- Set the auto_dw of the fixed layers to zeros
  -- for i=1,self.m_fixed_layers do
  --     -- self.network:get(self.fixed_layers[i]).gradWeight:zero()
  --     -- self.network:get(self.fixed_layers[i]).gradBias:zero()
  --     self.auto_network:get(self.fixed_layers[i]):zeroGradParameters()
  -- end

  -- accumulate update
  -- w += lr * dw ./ sqrt(g2 - g^2 +0.01)
  network.deltas:mul(0):addcdiv(-lr, network.dw, network.tmp) -- 0 + lr * dw ./ tmp
  network.w:add(network.deltas)

end

-- Train a discriminator for the ADDA approach
function adt:trainDiscriminator(s_sim, s_real, lr)
  -- Construct inputs
  local f
  if self.adversarial_mode == 1 then
    local f_sim = self.encoder_sr.network:forward(s_sim)
    local f_real = self.encoder_tg.network:forward(s_real)
    f = torch.cat(f_sim,f_real,1)
  elseif self.adversarial_mode == 2 then -- shared the weights in encoders
    local s = torch.cat(s_sim,s_real,1)
    f = self.encoder_tg.network:forward(s)
  else
    error('Please set which adversarial loss to use: 1:ADDA; 2:DC !!!')
  end

  if self.shortcut_mode then
    f = self.pose_net.network:forward(f)
  end

  -- Construct labels
  local true_pro = torch.ones(s_sim:size(1))
  local false_pro = torch.zeros(s_sim:size(1))
  local l_sim = torch.cat(true_pro, false_pro,2)
  local l_real = torch.cat(false_pro, true_pro,2)
  local l = torch.cat(l_sim,l_real,1)

  -- if self.gpu >= 0 then
  --   l = self:cuda(l)
  -- end
  local outputs = self.discriminater.network:forward(f):float()
  local delta = self:getDelta(outputs, l, self.criterion_KL)
  local loss = self:getLoss(outputs, l, self.criterion_KL)

  -- Optimize the discriminater network
  -- lr = lr * (1.7 - self.ad_weight / 0.02)
  -- print("Discriminater lr: ", lr)
  self:getGradient(f,delta,self.discriminater)
  self:AdamOptimizer(self.discriminater,self.d_lr_discount*lr)

  return loss
end

-- Get the componded gradients considering both simulated and real samples
function adt:getTaskGradientsMix(s_sim, s_real_t, l_sim, l_real_t)
  local p_gradients_cnn_real, p_gradients_fc_real = self:getTaskGradients(s_real_t, l_real_t, self.encoder_tg)
  
  local p_gradients_cnn_sim, p_gradients_fc_sim
  if self.parallel_mode then
    -- Clone the gradients to avoid overwriting by the gradient update in later procesures
    p_gradients_fc_real = p_gradients_fc_real:clone()
    p_gradients_cnn_sim, p_gradients_fc_sim = self:getTaskGradients(s_sim, l_sim, self.encoder_sr)
    
    -- Mix the fc gradients
    p_gradients_fc_sim:mul(0.5):add(0.5,p_gradients_fc_real)
    p_gradients_fc_real = p_gradients_fc_sim
  end
    
  return p_gradients_cnn_sim, p_gradients_cnn_real, p_gradients_fc_real

end

-- Get the gradients for target pose recognition
function adt:getTaskGradients(s, l, encoder)
  local s_labelled = s
  local l_labelled = l
  local f = encoder.network:forward(s_labelled)
  local outputs = self.pose_net.network:forward(f):float()
  local p_size = outputs:size(2)
  local l_size = l_labelled:size(2)
  if p_size > l_size then
    local l_temp = outputs:clone()
    l_temp:narrow(2,1,l_size):copy(l_labelled)
    l_labelled = l_temp
  end
  local delta0 = self:getDelta(outputs, l_labelled, self.criterion_MSE)
  if self.shortcut_mode then
    delta0 = delta0:mul(p_size):div(l_size)
  end
  local p_gradients_fc, delta1 = self:getGradient(f,delta0,self.pose_net)
  local p_gradients_cnn, _ = self:getGradient(s_labelled,delta1,encoder)

  return p_gradients_cnn, p_gradients_fc

end

-- Not compatible with shortcuts yet
function adt:getE2EGradients(s, l)

    -- Get pose estimation gradients
    -- local s_labelled = torch.cat(s_sim, s_real_t, 1)
    -- local l_labelled = torch.cat(l_sim, l_real_t, 1)
    -- local s_labelled = s_real_t
    -- local l_labelled = l_real_t
    local f = self.encoder_tg.network:forward(s[2])
    local p2 = self.pose_net.network:forward(f)
    -- if p2:size(2) > 3 then
    --   p2 = p2:narrow(2,1,3)
    -- end
    local p = torch.cat(s[1],p2,2)
    local outputs = self.ctrl_net.network:forward(p):float()
    local delta0 = self:getDelta(outputs, l, self.criterion_MSE)
    local ctrl_gradients, delta1 = self:getGradient(p,delta0,self.ctrl_net)
    delta1 = delta1:narrow(2,8,3)
    local p_gradients_fc, delta2 = self:getGradient(f,delta1,self.pose_net)
    local p_gradients_cnn, _ = self:getGradient(s[2],delta2,self.encoder_tg)

    return p_gradients_cnn, p_gradients_fc, ctrl_gradients

end

function adt:getAdversarialGradients(s_sim,  s_real)
  if self.adversarial_mode == 1 then
    return self:getADDAGradients(s_sim,  s_real)
  elseif self.adversarial_mode == 2 then
    return self:getDCGradients(s_sim,  s_real)
  else
    error('Please set which adversarial loss to use: 1:ADDA; 2:DC !!!')
  end
end


function adt:getADDAGradients(s_sim,  s_real)
  local s = s_real
  local f0 = self.encoder_tg.network:forward(s)
  local f = f0
  if self.shortcut_mode then
    f = self.pose_net.network:forward(f0)
  end
  local true_pro = torch.ones(f:size(1))
  local false_pro = torch.zeros(f:size(1))
  local l = torch.cat(true_pro, false_pro,2) -- fake sim

  local outputs = self.discriminater.network:forward(f):float()
  local delta = self:getDelta(outputs, l, self.criterion_KL)
  _, delta = self:getGradient(f, delta, self.discriminater)
  local d_gradients_fc
  if self.shortcut_mode then
    d_gradients_fc, delta = self:getGradient(f0,delta,self.pose_net)
  end
  local d_gradients_cnn, _ = self:getGradient(s,delta,self.encoder_tg)

  return d_gradients_cnn, d_gradients_fc
end


function adt:getDCGradients(s_sim,  s_real)
  -- local s = s_real
  -- local f0 = self.encoder_tg.network:forward(s)
  -- local f = f0
  -- if self.shortcut_mode then
  --   f = self.pose_net.network:forward(f0)
  -- end
  -- local true_pro = torch.ones(f:size(1))
  -- local false_pro = torch.zeros(f:size(1))
  -- local l = torch.cat(true_pro, false_pro,2) -- fake sim

  local s = torch.cat(s_sim,s_real,1)
  local f0 = self.encoder_tg.network:forward(s)
  local f = f0
  if self.shortcut_mode then
    f = self.pose_net.network:forward(f0)
  end
  local l = 0.5*torch.ones(f:size(1),2) -- uniform labells

  local outputs = self.discriminater.network:forward(f):float()
  local delta = self:getDelta(outputs, l, self.criterion_KL)
  _, delta = self:getGradient(f, delta, self.discriminater)
  local d_gradients_fc
  if self.shortcut_mode then
    d_gradients_fc, delta = self:getGradient(f0,delta,self.pose_net)
  end
  local d_gradients_cnn, _ = self:getGradient(s,delta,self.encoder_tg)

  return d_gradients_cnn, d_gradients_fc
end


function adt:getAdWeight_expPID(l_goal, l)
  self.e_d = l_goal - l  -- the error between goal loss and current loss
  self.integral_e_d = self.integral_e_d + self.e_d  -- the integral of errors, assuming dt = 1
  -- Limit integral to solve windup problems
  if self.integral_e_d > 12.5 then
    self.integral_e_d = 12.5
  elseif self.integral_e_d < -12.5 then
    self.integral_e_d = -12.5
  end
  self.derivative_e_d = self.e_d - self.e_d_previous  -- the derivative of errors, assuming dt = 1
  self.e_d_previous = self.e_d

  local output_PID = self.Kp*self.e_d + self.Ki*self.integral_e_d + self.Kd*self.derivative_e_d

  -- self.PID_map_scalar/(1+exp(-output_PID*50)
  local AdWeight = math.exp(50*(-output_PID))
  AdWeight = self.PID_map_scalar / (1 + AdWeight)
  -- AdWeight = 0.02 / (1 + AdWeight)

  return AdWeight
end

-- Train a target encoder in an unsupervised manner using only the adversarial loss (ADDA)
function adt:trainEncoder_adversarial(s_sim, s_real_t, s_real, l_sim, l_real_t, lr, loss)
  -- Discounted learning rate
  lr = lr * self.d_lr_discount

  -- Get ADDA Gradients
  if self.ad_weight > 0 then
    local d_gradients_cnn, d_gradients_fc = self:getAdversarialGradients(s_sim,  s_real)

    if self.shortcut_mode then
      self:AdamOptimizer(self.pose_net, self.ad_weight*lr)
    end
    self:AdamOptimizer(self.encoder_tg, self.ad_weight*lr)
  end

end

-- Train a target encoder in a semi-supervised manner, i.e., adversarial + supervised losses
function adt:trainEncoder_semi(s_sim, s_real_t, s_real, l_sim, l_real_t, lr, loss)

  -- Get Task Gradients
  local _, p_gradients_cnn, p_gradients_fc = self:getTaskGradientsMix(s_sim, s_real_t, l_sim, l_real_t)
  -- Clone the gradients to avoid overwriting by the gradient update in later procesures
  if self.shortcut_mode then
    p_gradients_fc = p_gradients_fc:clone()
  end
  p_gradients_cnn = p_gradients_cnn:clone()

  -- Get ADDA Gradients
  if self.ad_weight > 0 then
    local d_gradients_cnn, d_gradients_fc = self:getAdversarialGradients(s_sim,  s_real)

    -- Fuse Gradients
    if self.shortcut_mode then
      d_gradients_fc:mul(self.ad_weight*self.d_lr_discount):add(self.task_weight,p_gradients_fc) -- merge the dw from two losses (discriminater and pose estimation)
    end
    d_gradients_cnn:mul(self.ad_weight*self.d_lr_discount):add(self.task_weight,p_gradients_cnn) -- merge the dw from two losses (discriminater and pose estimation)
  end

  -- Update relevant modules using Adam
  self:AdamOptimizer(self.pose_net, lr)
  self:AdamOptimizer(self.encoder_tg, lr)
  if self.parallel_mode then
    self:AdamOptimizer(self.encoder_sr, lr)
  end

end

function adt:trainE2E_semi(s_sim, s_real_t, s_real, l_sim, l_real_t, lr, loss, s, l, p)

  -- Put the samples for end-to-end training in perception update
  s_real_t = torch.cat(s_real_t, s[2], 1)
  l_real_t = torch.cat(l_real_t, p, 1)

  -- Put the samples for end-to-end training in the calculation for ad loss and also perception task loss
  s_sim = torch.cat(s_sim, s[2], 1)

  -- Get Task Gradients
  local _, p_gradients_cnn, p_gradients_fc = self:getTaskGradientsMix(s_sim, s_real_t, l_sim, l_real_t)
  -- Clone the gradients to avoid the overwriting by the gradient update by later procesures
  p_gradients_fc = p_gradients_fc:clone()
  p_gradients_cnn = p_gradients_cnn:clone()

  -- Get ADDA Gradients
  if self.ad_weight > 0 then
    local d_gradients_cnn, d_gradients_fc = self:getAdversarialGradients(s_sim,  s_real)
    d_gradients_cnn = d_gradients_cnn:clone()

    -- Fuse Gradients
    if self.shortcut_mode then
      d_gradients_fc = d_gradients_fc:clone()
      p_gradients_fc:mul(self.task_weight):add(self.ad_weight*self.d_lr_discount,d_gradients_fc) -- merge the dw from two losses (discriminater and pose estimation)
    end
    p_gradients_cnn:mul(self.task_weight):add(self.ad_weight*self.d_lr_discount,d_gradients_cnn) -- merge the dw from two losses (discriminater and pose estimation)

  end

  -- Get end-to-end control Gradients
  local e2e_gradients_encoder, e2e_gradients_posenet, e2e_gradients_ctrl = self:getE2EGradients(s, l)

  e2e_gradients_encoder:mul(self.c_scalar):add(self.p_scalar,p_gradients_cnn)
  -- if self.shortcut_mode then
    -- e2e_gradients_posenet:mul(self.c_scalar):add(self.p_scalar,d_gradients_fc)
  -- else
  e2e_gradients_posenet:mul(self.c_scalar):add(self.p_scalar,p_gradients_fc)
  -- end

  -- Update relevant modules using Adam
  self:AdamOptimizer(self.ctrl_net, lr)
  self:AdamOptimizer(self.pose_net, lr)
  self:AdamOptimizer(self.encoder_tg, lr)
  if self.parallel_mode then
    self:AdamOptimizer(self.encoder_sr, lr)
  end

end

-- Train a target encoder
function adt:trainEncoder(s_sim, s_real_t, s_real, l_sim, l_real_t, lr, loss)

  -- Update current smart dc weight
  if self.use_PID then
    table.insert(self.latest_n_d_loss,loss)
    if #self.latest_n_d_loss > self.d_loss_filter_length then
      table.remove(self.latest_n_d_loss,1)
      -- self.ad_weight = torch.mean(torch.Tensor(self.latest_n_d_loss))
      -- self.ad_weight = torch.max(torch.Tensor(self.latest_n_d_loss))
      self.ad_weight, _ = torch.median(torch.Tensor(self.latest_n_d_loss))
      self.ad_weight = self.ad_weight[1]

      self.ad_weight = self:getAdWeight_expPID(self.desired_d_loss, self.ad_weight)
      print("current Ad weight: ",self.ad_weight)

      -- self.task_weight = 1 - self.ad_weight
      -- self.task_weight = 1

      -- print("Current DC Weight: ", self.ad_weight)
    end
  end

  if self.semi_supervised then
    self:trainEncoder_semi(s_sim, s_real_t, s_real, l_sim, l_real_t, lr, loss)
  else
    self:trainEncoder_adversarial(s_sim, s_real_t, s_real, l_sim, l_real_t, lr, loss)
  end

end

-- End-to-end training function
function adt:trainE2E(s_sim, s_real_t, s_real, l_sim, l_real_t, lr, loss, s, l, p)

  -- Update current smart dc weight
  if self.use_PID then
    table.insert(self.latest_n_d_loss,loss)
    if #self.latest_n_d_loss > self.d_loss_filter_length then
      table.remove(self.latest_n_d_loss,1)
      -- self.ad_weight = torch.mean(torch.Tensor(self.latest_n_d_loss))
      -- self.ad_weight = torch.max(torch.Tensor(self.latest_n_d_loss))
      self.ad_weight, _ = torch.median(torch.Tensor(self.latest_n_d_loss))
      self.ad_weight = self.ad_weight[1]

      self.ad_weight = self:getAdWeight_expPID(self.desired_d_loss, self.ad_weight)
      print("current Ad weight: ",self.ad_weight)

      -- self.task_weight = 1 - self.ad_weight
      -- self.task_weight = 1

      -- print("Current DC Weight: ", self.ad_weight)
    end
  end

  -- if self.semi_supervised then
  self:trainE2E_semi(s_sim, s_real_t, s_real, l_sim, l_real_t, lr, loss, s, l, p)
  -- else
    -- self:trainE2E_adversarial(s_sim, s_real_t, s_real, l_sim, l_real_t, lr, loss, s, l, p)
  -- end
end

-- Learning in minibatch manner
function adt:learnMinibatch(s_sim, s_real_t, s_real, l_sim, l_real_t, s, l, p)

    if self.gpu >= 0 then
      s_sim = self:cuda(s_sim)
      s_real_t = self:cuda(s_real_t)
      s_real = self:cuda(s_real)
      if s then
        s = self:cuda(s)
      end
    end

    -- compute linearly annealed learning rate
    local t = math.max(0, self.numSteps - self.lr_startt)
    self.lr = (self.lr_start - self.lr_end) * (self.lr_endt - t)/self.lr_endt +
                self.lr_end
    self.lr = math.max(self.lr, self.lr_end)

    local loss = self:trainDiscriminator(s_sim, s_real, self.lr)
    print("D Loss: ", loss)
    if self.numSteps > self.encoder_start and self.numSteps%self.d_episode_steps < self.e_episode_steps then
      if s then
        self:trainE2E(s_sim, s_real_t, s_real, l_sim, l_real_t, self.lr, loss, s, l, p)
      else
        self:trainEncoder(s_sim, s_real_t, s_real, l_sim, l_real_t, self.lr, loss)
      end
    end

end

-- Validation set sampling
function adt:sample_validation_data(s_sim, s_real, pose_sim, pose_real, s, l)
    s_sim = self:float(self:preprocess(s_sim))
    s_real = self:float(self:preprocess(s_real))
    if s then
      s = self:float(self:preprocess(s))
    end
    self.validation_s_sim = s_sim
    self.validation_s_real = s_real
    self.validation_pose_sim = pose_sim
    self.validation_pose_real = pose_real
    if s then
      self.validation_s = s
      self.validation_l = l
    end

    if self.gpu >= 0 then
      self.validation_s_sim = self:cuda(self.validation_s_sim)
      self.validation_s_real = self:cuda(self.validation_s_real)
      if s then
        self.validation_s = self:cuda(self.validation_s)
      end

    end
end

-- Validation function
-- TODO: sort out the code, summarize the function with some formula in comments
function adt:compute_validation_statistics(s_sim, s_real, pose_sim, pose_real, e2e_s, e2e_l)

    if not s_sim then
      s_sim = self.validation_s_sim
      s_real = self.validation_s_real
      pose_sim = self.validation_pose_sim
      pose_real = self.validation_pose_real
      if self.validation_s then
        e2e_s = self.validation_s
        e2e_l = self.validation_l
      end

    end

    -- local s = torch.cat(s_sim,s_real,1)
    -- local f = self.encoder_tg.network:forward(s)
    -- -- f = self.pose_net.network:forward(f)

    local f_sim = self.encoder_sr.network:forward(s_sim)
    local f_real = self.encoder_tg.network:forward(s_real)
    local f = torch.cat(f_sim,f_real,1)
    if self.shortcut_mode then
      f = self.pose_net.network:forward(f)
    end

    local true_pro = torch.ones(s_sim:size(1))
    local false_pro = torch.zeros(s_sim:size(1))
    local l_sim = torch.cat(true_pro, false_pro,2)
    local l_real = torch.cat(false_pro, true_pro,2)
    local l = torch.cat(l_sim,l_real,1)

    local d_batch_size = f:size(1)
    local p_batch_size = s_real:size(1)

    -- Compute predictions
    local d_p = self.discriminater.network:forward(f):float()
    -- Cross Entropy Loss
    local d_loss = self.criterion_KL:forward(d_p, l)
    print("d loss: ", d_loss)
    -- d_loss = d_loss / d_batch_size
    local d_p_conf, d_p_index = torch.max(d_p,2)
    local l_conf, l_index = torch.max(l,2)
    -- print("Prediction: ", d_p_index)
    local correctness = torch.eq(d_p_index:byte(),l_index:byte())
    local accuracy = torch.sum(correctness)/d_batch_size
    print("D Accuracy: ", accuracy)
    print("Confidence: ", d_p_conf)
    -- print("Correctness: ", correctness)

    -- Real Perception Loss
    local f_real = self.encoder_tg.network:forward(s_real)
    local outputs = self.pose_net.network:forward(f_real):float()
    if outputs:size(2) > 3 then
      outputs = outputs:narrow(2,1,3)
    end
    local delta = self:getDelta(outputs, pose_real, self.criterion_MSE, true)
    -- local real_loss = self:getLoss(outputs, pose_real, self.criterion_MSE)
    -- print("loss: ", real_loss)
    local a = delta:nElement() / 2
    -- The scalar a is to compensate norm which is multiplied by in the criterion_MSE.backward process
    -- real norm = (sizeAverage ? 2./((real)THTensor_(nElement)(input)) : 2.);

    -- cost_i = sum((y_i - target_i) ^ 2) / batch_size,
    -- where i represents the output index of an output vector
    delta = delta:float():mul(a)
    delta:pow(2)
    local cost = torch.sum(delta,1):div(p_batch_size)
    local cost_sum = torch.sum(cost)

    -- Sim Perception Loss
    -- cost_i = sum((y_i - target_i) ^ 2) / batch_size
    local f_sim = self.encoder_sr.network:forward(s_sim)
    outputs = self.pose_net.network:forward(f_sim):float()
    if outputs:size(2) > 3 then
      outputs = outputs:narrow(2,1,3)
    end
    local delta_sim = self:getDelta(outputs, pose_sim, self.criterion_MSE, true)
    delta_sim = delta_sim:float():mul(a)
    delta_sim:pow(2)
    local cost_sim = torch.sum(delta_sim,1):div(p_batch_size)
    local cost_sum_sim = torch.sum(cost_sim)

    -- E2E loss
    -- cost_i = sum((y_i - target_i) ^ 2) / batch_size
    local e2e_cost, e2e_cost_sum
    if e2e_s then
      f = self.encoder_tg.network:forward(e2e_s[2])
      local p2 = self.pose_net.network:forward(f)
      -- print(p2:size())
      local p = torch.cat(e2e_s[1],p2,2)
      -- print(p:size())
      outputs = self.ctrl_net.network:forward(p):float()
      -- print(outputs:size())

      local e2e_batch_size = e2e_l:size(1)
      -- print(e2e_l:size())

      local delta_e2e = self:getDelta(outputs, e2e_l, self.criterion_MSE, true)
      local a = delta_e2e:nElement() / 2
      delta_e2e = delta_e2e:float():mul(a)
      delta_e2e:pow(2)
      e2e_cost = torch.sum(delta_e2e,1):div(e2e_batch_size)
      e2e_cost_sum = torch.sum(e2e_cost)
    end

    -- Update the cost histories
    local index = #self.hist_Lp_sum_real + 1
    self.hist_Lp_vector_real[index] = cost
    self.hist_Lp_sum_real[index] = cost_sum
    self.hist_Lp_vector_sim[index] = cost_sim
    self.hist_Lp_sum_sim[index] = cost_sum_sim
    self.hist_LD[index] = d_loss
    if e2e_cost then
      self.hist_Le2e_vector_sim[index] = e2e_cost
      self.hist_Le2e_sum_sim[index] = e2e_cost_sum
    end

    return cost, cost_sum, d_loss, cost_sum_sim, e2e_cost, e2e_cost_sum
end


function adt:train(s_sim, s_real_t, s_real, l_sim, l_real_t, s, l, p)
    -- Preprocess state (will be set to nil if terminal)
    -- print(#s)
    -- s = torch.Tensor(s)
    -- print(s:size(1))
    s_sim = self:float(self:preprocess(s_sim))
    s_real = self:float(self:preprocess(s_real))
    s_real_t = self:float(self:preprocess(s_real_t))
    if s then
      s = self:float(self:preprocess(s))
    end

    -- Do the learning
    self.encoder_tg.network:training()
    self.encoder_sr.network:training()
    self.pose_net.network:training()
    self.discriminater.network:training()
    if self.mode ~= 2 then -- not in perception mode
      self.ctrl_net.network:training()
    end
    for i = 1, self.n_replay do
      self:learnMinibatch(s_sim, s_real_t, s_real, l_sim, l_real_t, s, l, p)
    end
    self.encoder_tg.network:evaluate()
    self.encoder_sr.network:evaluate()
    self.pose_net.network:evaluate()
    self.discriminater.network:evaluate()
    if self.mode ~= 2 then -- not in perception mode
      self.ctrl_net.network:evaluate()
    end

    self.numSteps = self.numSteps + 1
end


function adt:test(s)
    -- Preprocess state (will be set to nil if terminal)
    -- print(#s)
    -- s = torch.Tensor(s)
    s = self:float(self:preprocess(s))

    if self.gpu >= 0 then s = self:cuda(s) end

    -- Compute object pose predictions
    local s_im = s
    if type(s) == 'table' then -- Compatible to both perception and e2e modes
      s_im = s[2]
    end
    local p = self.encoder_tg.network:forward(s_im)
    p = self.pose_net.network:forward(p)
    -- Pick the 3 DoF pose info, compatible to single frame and batches
    if p:size(p:dim()) > 3 then
      p = p:narrow(p:dim(),1,3)
    end

    -- Compute joint velocity predictions
    local v = p
    if self.mode == 1 and type(s) == 'table' then -- e2e mode and s is a table
      p = torch.cat(s[1], p, 1)
      v = self.ctrl_net.network:forward(p):float()
    end

    return v
end


function adt:createNetwork()
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

function adt:report()
    print("========================== Target Encoder ===========================")
    print(get_weight_norms(self.encoder_tg.network))
    print(get_grad_norms(self.encoder_tg.network))
    print("========================== Discriminater ===========================")
    print(get_weight_norms(self.discriminater.network))
    print(get_grad_norms(self.discriminater.network))
end

-- Change Tensor or Tensors in a table into float type.
function adt:float(s)
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
function adt:cuda(s)
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
