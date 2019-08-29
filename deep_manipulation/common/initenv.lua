--[[ File
@description:
    This class is for system initialization.
@version: V0.01
@author: Fangyi Zhang   email:gzzhangfangyi@gmail.com
@acknowledgement:
    ARC Centre of Excellence for Robotic Vision (ACRV)
    Queensland Univsersity of Technology (QUT)
@history:
    V0.00   03/06/2016  developed the first version
    V0.01   30/05/2017  enabled the setting for actions through external parameters
]]

dmn = {}

require 'torch'
require 'nn'
require 'nngraph'
require 'net/nnutils'
require 'net/Scale'
require 'net/Rectifier'
require 'learner/SupervisedLearner'
require 'learner/SupervisedFinetuner'
require 'learner/ADTLearner'
require 'image'
package.cpath = package.cpath .. ";./external_lib/?.so" -- Include external libs

-- Modified by Fangyi Zhang
-- to separate the learning core from Atari game engine and interface
function torchSetup(_opt)
    _opt = _opt or {}
    local opt = table.copy(_opt)
    assert(opt)

    -- preprocess options:
    --- convert options strings to tables
    if opt.agent_params then
        opt.agent_params = str_to_table(opt.agent_params)
        opt.agent_params.gpu       = opt.gpu
        opt.agent_params.best      = opt.best
        opt.agent_params.verbose   = opt.verbose
        if opt.network ~= '' then
            opt.agent_params.network = opt.network
        end
        if opt.p_network ~= '' then
            opt.agent_params.p_network = opt.p_network
        end
        if opt.real_dataset ~= '' then
            opt.agent_params.real_dataset = opt.real_dataset
        end
    end

    --- general setup
    opt.tensorType =  opt.tensorType or 'torch.FloatTensor'
    torch.setdefaulttensortype(opt.tensorType) -- Set the default tensor type
    if not opt.threads then
        opt.threads = 4
    end
    torch.setnumthreads(opt.threads)
    if not opt.verbose then
        opt.verbose = 10
    end
    if opt.verbose >= 1 then
        print('Torch Threads:', torch.getnumthreads())
    end

    --- set gpu device
    if opt.gpu and opt.gpu >= 0 then
        require 'cutorch'
        require 'cunn'
        if opt.gpu == 0 then
            local gpu_id = tonumber(os.getenv('GPU_ID'))
            if gpu_id then opt.gpu = gpu_id+1 end
        end
        if opt.gpu > 0 then cutorch.setDevice(opt.gpu) end
        opt.gpu = cutorch.getDevice()
        print('Using GPU device id:', opt.gpu-1)
    else
        opt.gpu = -1
        if opt.verbose >= 1 then
            print('Using CPU code only. GPU device id:', opt.gpu)
        end
    end

    --- set up random number generators
    -- removing lua RNG; seeding torch RNG with opt.seed and setting cutorch
    -- RNG seed to the first uniform random int32 from the previous RNG;
    -- this is preferred because using the same seed for both generators
    -- may introduce correlations; we assume that both torch RNGs ensure
    -- adequate dispersion for different seeds.
    math.random = nil
    opt.seed = opt.seed or 1
    torch.manualSeed(opt.seed)
    if opt.verbose >= 1 then
        print('Torch Seed:', torch.initialSeed())
    end
    local firstRandInt = torch.random()
    if opt.gpu >= 0 then
        cutorch.manualSeed(firstRandInt)
        if opt.verbose >= 1 then
            print('CUTorch Seed:', cutorch.initialSeed())
        end
    end

    return opt
end

-- Modified by Fangyi Zhang
-- to separate the learning core from Atari game engine and interface
-- and to connect to new simulation environments, lua-sim and matlab-sim
function setup(_opt)
    assert(_opt)

    --preprocess options:
    --- convert options strings to tables
    _opt.agent_params = str_to_table(_opt.agent_params)
    if _opt.agent_params.transition_params then
        _opt.agent_params.transition_params =
            str_to_table(_opt.agent_params.transition_params)
    end

    --- first things first
    local opt = torchSetup(_opt)

    local gameActions = {}
    local n_actions = _opt.agent_params.n_actions or 15
    for i=1,n_actions do -- 7dof
      gameActions[i] = i
    end

    -- agent options
    _opt.agent_params.actions   = gameActions
    _opt.agent_params.gpu       = _opt.gpu
    _opt.agent_params.best      = _opt.best
    -- If a t7 pre-trained network is provided, the network settings in agent_params will be replaced.
    if _opt.network ~= '' then
        _opt.agent_params.network = _opt.network
    end
    if _opt.p_network ~= '' then
        _opt.agent_params.p_network = _opt.p_network
    end
    if _opt.real_dataset ~= '' then
        _opt.agent_params.real_dataset = _opt.real_dataset
    end
    _opt.agent_params.verbose = _opt.verbose

    local agent = dmn[_opt.agent](_opt.agent_params)

    if opt.verbose >= 1 then
        print('Set up Torch using these options:')
        for k, v in pairs(opt) do
            print(k, v)
        end
    end

    -- return gameActions, agent, opt
    return gameActions, agent, opt
end



--- other functions

function str_to_table(str)
    if type(str) == 'table' then
        return str
    end
    if not str or type(str) ~= 'string' then
        if type(str) == 'table' then
            return str
        end
        return {}
    end
    local ttr
    if str ~= '' then
        local ttx=tt
        loadstring('tt = {' .. str .. '}')()
        ttr = tt
        tt = ttx
    else
        ttr = {}
    end
    return ttr
end

function table.copy(t)
    if t == nil then return nil end
    local nt = {}
    for k, v in pairs(t) do
        if type(v) == 'table' then
            nt[k] = table.copy(v)
        else
            nt[k] = v
        end
    end
    setmetatable(nt, table.copy(getmetatable(t)))
    return nt
end
