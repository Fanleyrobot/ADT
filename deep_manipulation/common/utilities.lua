--[[ File
@description:
    Utility functions for lua table and dataset and model manipulation
@version: V0.10
@author: Fangyi Zhang   email:gzzhangfangyi@gmail.com
@acknowledgement:
    ARC Centre of Excellence for Robotic Vision (ACRV)
    Queensland Univsersity of Technology (QUT)
@history:
    V0.00   14/06/2017  developed the first version
    V0.10   26/07/2018  minor updates for code re-organization
]]


require 'torch'
require 'image'


-- A function to load a dataset
-- TODO: To modify with modifications with simulator and learner classes
-- Define a consistent format for datasets and relevant conversion fuctions
function load_pre_constructed_dataset(file)
    -- try to load the dataset
    local err_msg,  dataset= pcall(torch.load, file)
    if not err_msg then
        error("------!!! Could not find the dataset file !!!------")
    end
    return dataset
end

-- A function to load t7 file
function load_t7_File(file_name)
    -- try to load the dataset
    local err_msg, data = pcall(torch.load, file_name)
    if not err_msg then
        error("------!!! Could not find the t7 file !!!------")
    end
    return data
end

-- A function to convert a cuda model to a cpu model
function CudaNN_To_CpuNN(cu_model)
  -- transform it to CPU network
  local cpu_model = cu_model:clone():float()

  return cpu_model
end

-- A function to deeply copy a table
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

-- A function to merge two tables
function table.merge(t1,t2)
  -- local t = table.copy(t1)
  local t = t1
  for k, v in ipairs(t2) do
    table.insert(t, v)
  end
  return t
end

-- Recover normalized low-dim features to their original values
-- TODO: To update with the modifications of the simulator classes
function recoverLowDimTensor(state)
    local target_pose_max = {0.85, 0.6, 0.5708}
    local target_pose_min = {0.35, 0.0, 0.3908}
    local state = state:clone()

    local n = state:size(1)
    -- Recover destination pose features
    for i=1, n do
        state[i] = state[i] * (target_pose_max[i] - target_pose_min[i]) + target_pose_min[i]
    end

    return state
end

-- A function to merge multiple datasets into one
-- TODO: To update with the modifications of the simulator classes
function merge_dataset(name, start_index, end_index)

  local merged_data
  -- local first_amount
  for i=start_index,end_index do
    local dataset_name = name .. i .. '.t7'
    local new_data = load_pre_constructed_dataset(dataset_name)
    print("Dataset Loaded: ", dataset_name)

    -- -- Process the raw data
    -- -- print(new_data['low_dim_s'][1]:narrow(1,8,3))
    -- local target = recoverLowDimTensor(new_data['low_dim_s'][1]:narrow(1,8,3))
    -- -- print(target)
    -- target = target:totable()
    -- new_data['object_position'] = {}
    -- for j=1,new_data['sample_amount'] do
    --     new_data['object_position'][j] = table.copy(target)
    --     -- new_data['object_position'][j] = table.copy(end_effect
    -- end
    -- -- print(new_data['object_position'][new_data['sample_amount']])
    -- print(torch.Tensor(new_data['object_position'][new_data['sample_amount']]))
    -- new_data['vel'] = new_data['vel_cmd']
    -- -- print("Vel: ", #new_data['vel'])
    -- -- print("Vel: ", new_data['vel'][new_data['sample_amount']])
    -- print("Vel: ", torch.Tensor(new_data['vel'][new_data['sample_amount']]))

    if not merged_data then
      print("======== Added the First Dataset ========")
      merged_data = new_data
      -- first_amount = merged_data['sample_amount']
    else
      print("======== Merging Dataset: ", i, " ========")
      for k,v in pairs(new_data) do
        print("Element Merged: ", k)
        if type(v) == 'table' then
          merged_data[k] = table.merge(merged_data[k], v)
        elseif k == 'sample_amount' then
          merged_data[k] = merged_data[k] + new_data[k]
        end
      end
    end
    new_data = nil
    collectgarbage()


    -- object_pose = target_pose_[-1],
    -- end_effector_pose = target_pose_e[-1],
    -- desired_pose = desired_pose_[-1],
    -- init_pose = initial_arm_pose_[-1],

    -- if i==end_index then
    --   print(merged_data['low_dim_s_e'][first_amount+547])
    --   print(new_data['low_dim_s'][547])
    --   print(merged_data['low_dim_s_e'][first_amount+43])
    --   print(new_data['low_dim_s'][43])
    --   print(merged_data['low_dim_s_e'][first_amount+12784])
    --   print(new_data['low_dim_s'][12784])
    -- end
  end
  print("<<<<<< Merged Data Amount: ", merged_data['sample_amount'], " >>>>>>")
  -- print("real amount s e: ", #merged_data['low_dim_s_e'])
  -- print("real amount s: ", #merged_data['low_dim_s'])
  -- print("real amount l: ", #merged_data['label'])

  -- merged_data['image400'] = nil
  -- merged_data['image300'] = nil

  return merged_data

end

-- A function to replace images in a dataset
-- TODO: To update with the modifications of the simulator classes
function replace_dataset_image(name, path)
  local dataset_name = name .. '.t7'
  local origin_data = load_pre_constructed_dataset(dataset_name)
  print("Dataset Loaded: ", dataset_name)
  local sample_amount = origin_data.sample_amount

  -- local first_amount
  for i=1,sample_amount do
    local imagefile = path .. i .. '.png'
    local img = image.load(imagefile,3,'float')
    print("pixel max: ", img:max())
    print("pixel min: ", img:min())
    origin_data.image[i] = img
  end

  collectgarbage()
  print("<<<<<< Dataset Updated >>>>>>")
  -- print("real amount s e: ", #merged_data['low_dim_s_e'])
  -- print("real amount s: ", #merged_data['low_dim_s'])
  -- print("real amount l: ", #merged_data['label'])

  -- merged_data['image400'] = nil
  -- merged_data['image300'] = nil

  return origin_data

end
