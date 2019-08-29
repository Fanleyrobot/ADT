--[[ File
@description:
    The functions included in this file are for visualization.
@version: V0.18
@author: Fangyi Zhang   email:gzzhangfangyi@gmail.com
@acknowledgement:
    ARC Centre of Excellence for Robotic Vision (ACRV)
    Queensland Univsersity of Technology (QUT)
@history:
    V0.00   07/08/2015  developed the first version
    V0.10   07/08/2015  updated to support setting to save to files or not
    V0.12   18/11/2015  adapted to the new version of network visualization code
    V0.13   12/05/2016  added the setup for visu_output and visu_weight
    V0.15   19/05/2016  added the function of autonomously initialize simage_h, simage_w and w_layers.
    V0.16   20/05/2016  changed the weight visualization from every frame to only one frame during a period.
    V0.17   29/08/2017  changed the image.display setting to visualize without minmax operation
]]

require 'torch'
lfs = require 'lfs' -- for operating the file system, i.e., getting the current working directory
require 'image'
require 'common/network_visualization'
require 'common/FramesToVideo' -- for generating videos from image frames


-- construct a class
local visu = torch.class('visualization')


--[[ Function
@description: initialize a visualization object
@input:
    args: settings for a visualization object, i.e., {network=agent.network, options=opt, print_action_value=true}
@output: nil
@notes:
]]
function visu:__init(args)

    self.network = args.network -- a network container
    self.input_dims = args.input_dims or {84,84}-- Set input dims

    -- Initialize a network visulization object
    -- self.simage_h = args.simage_h or {0, 0, 0, 0, 0, 0, 0, 28, 16, 16, 9}
    -- self.simage_w = args.simage_w or {0, 0, 0, 0, 0, 0, 0, 28, 8, 8, 1}

    -- self.simage_h = args.simage_h or {16, 16, 16, 9}
    -- self.simage_w = args.simage_w or {1, 8, 8, 1}

    -- self.simage_h = args.simage_h or {12, 20, 20, 20, 20, 9}
    -- self.simage_w = args.simage_w or {1, 20, 20, 15, 15, 1}

    -- Settings for weights visualization
    -- Set the layers for weight visualization
    -- self.w_layers = args.w_layers or {2, 4, 6, 9, 11}

    -- self.w_layers = args.w_layers or {2, 4}

    -- self.w_layers = args.w_layers or {2, 4, 6}

    -- self.visu_output = args.visu_output or true -- whether visualize outputs
    self.visu_weight = args.visu_weight or false -- whether visualize weights

    self.visu_ob = network_visualization{network=self.network,
                                        network_type=args.options.gpu,
                                        input_dims=self.input_dims}
                                        -- simage_h=self.simage_h,
                                        -- simage_w=self.simage_w,
                                        -- w_layers=self.w_layers,
                                        -- visu_output=self.visu_output,
                                        -- visu_weight=self.visu_weight}
    self.num_bit = 9 -- the number bit when saving images for an easier video generation
    self.num_offset = math.pow(10, self.num_bit) -- the number offset when saving images for an easier video generation
    self.folder_volum = 1000 -- the maximum number of images contained in a folder

    -- Make a new folder to store simulation images and data
    local current_dir = lfs.currentdir()
    local data_parent_dir = current_dir .. "/running_data"
    lfs.mkdir(data_parent_dir)
    self.data_current_dir = data_parent_dir .. "/" .. args.options.plot_filename .. os.time()
    lfs.mkdir(self.data_current_dir)
    self.screen_dir =  self.data_current_dir .. "/screen_shots"
    lfs.mkdir(self.screen_dir)
    self.nn_outputs_dir = self.data_current_dir .. "/net_outputs_and_weights"
    lfs.mkdir(self.nn_outputs_dir)

    -- Temp directory path for making a folder for saving images in some certain epochs
    self.screen_dir_temp = self.screen_dir
    self.nn_outputs_dir_temp = self.nn_outputs_dir
    self.dir_screen_step = 0
    self.dir_nn_outputs_step = 0
    self.last_screen_saving_step = 0
    self.last_nn_outputs_saving_step = 0

    -- intialize the nn history variable
    self.nn_output_history = {}
    self.nn_output_sub_history = {}
    -- self.nn_weight_history = {}
    -- self.nn_weight_sub_history = {}

    -- Enable display when display and display available are set to true
    if (args.options.display_avail == 1) then
        if (args.options.display == 1) then
            self.display = true
        end
    end
    self.save_datafile = args.save_datafile or false -- whether to save t7 data files
    self.print_action_value = args.print_action_value or false -- whether print out the outputs of the last layer in each step

end


--[[ Function
@description: visualize screenshots
@input:
    screen: the current screen
    step: the current running step
    storage: whether to store images and videos
@output: nil
@notes:
]]
function visu:screen_shots(screen, step, storage)
    local store = storage or false

    if store then
        -- initial a saving folder
        if self.dir_screen_step == 0 then
            self.screen_dir_temp = self.screen_dir .. "/" .. step .. "-"
            lfs.mkdir(self.screen_dir_temp)
            self.dir_screen_step = step
        elseif step - self.dir_screen_step >= self.folder_volum or step - self.last_screen_saving_step > 1 then
            -- generate the video for the previous folder
            local frame_name = 'screenshots_1%' .. self.num_bit .. 'd.png'
            local video_name = 'screenshots_' .. self.dir_screen_step .. '-' .. self.last_screen_saving_step .. '.mp4'
            local ifile = self.screen_dir_temp .. '/' .. frame_name
            local vfile = self.screen_dir .. '/' .. video_name
            GenerateVideo(ifile, vfile, self.dir_screen_step)
            -- rename the previous folder
            -- os.execute("mv " .. self.screen_dir_temp .. " " .. self.screen_dir_temp .. self.last_screen_saving_step)

            -- delete the previous folder
            os.execute("rm -r " .. self.screen_dir_temp)

            -- create a new folder
            self.screen_dir_temp = self.screen_dir .. "/" .. step .. "-"
            lfs.mkdir(self.screen_dir_temp)
            self.dir_screen_step = step
        end

        image.save(self.screen_dir_temp .. '/screenshots_' .. self.num_offset+step .. '.png', screen)
        self.last_screen_saving_step = step
    end

    -- Show the screen when display available and display enabled
    if self.display then
        win_input = image.display{image=screen, win=win_input, min=0,max=1}
    end
end


--[[ Function
@description: visualize the output of each layer in a network
@input:
    step: the current running step
    storage: whether to store images, videos and data files
@output: nil
@notes:
]]
function visu:nn_outputs_and_weights(step, storage)
    local store = storage or false
    local merged_img_output, merged_img_weight
    local output_group, weight_group, output_sub_group, weight_sub_group

    if self.display or store then
        -- visualize network outputs
        merged_img_output, output_group, output_sub_group = self.visu_ob:visualize_outputs(step)
        -- print out the outputs of the last layer
        if self.print_action_value then
            print("Outputs of the last layer: (step "..step..")\n",output_group[11])
        end
    end

    if store then
        -- initial a saving folder
        if self.dir_nn_outputs_step == 0 then
            self.nn_outputs_dir_temp = self.nn_outputs_dir .. "/" .. step .. "-"
            lfs.mkdir(self.nn_outputs_dir_temp)
            self.dir_nn_outputs_step = step
        elseif step - self.dir_nn_outputs_step >= self.folder_volum or step - self.last_nn_outputs_saving_step > 1 then
            -- generate the video for the previous folder
            local frame_name = 'nn_data_1%' .. self.num_bit .. 'd.png'
            local video_name = 'nn_outputs_' .. self.dir_nn_outputs_step .. '-' .. self.last_nn_outputs_saving_step .. '.mp4'
            local ifile = self.nn_outputs_dir_temp .. '/' .. frame_name
            local vfile = self.nn_outputs_dir .. '/' .. video_name
            GenerateVideo(ifile, vfile, self.dir_nn_outputs_step)

            -- rename the previous folder
            --os.execute("mv " .. self.nn_outputs_dir_temp .. " " .. self.nn_outputs_dir_temp .. self.last_nn_outputs_saving_step)

            -- delete the previous folder
            os.execute("rm -r " .. self.nn_outputs_dir_temp)
            if self.visu_weight then
                merged_img_weight, weight_group, weight_sub_group = self.visu_ob:visualize_weights(step)
                image.save(self.nn_outputs_dir.."/nn_weights_"..self.dir_nn_outputs_step..'-'..self.last_nn_outputs_saving_step..".png", merged_img_weight)
            end
            -- save the neural network data in the previous period in a data file
            if self.save_datafile then
                local data_filename = self.nn_outputs_dir .. '/' .. 'nn_data_' .. self.dir_nn_outputs_step .. '-' .. self.last_nn_outputs_saving_step .. '.t7'
                torch.save(data_filename, {start_step=self.dir_nn_outputs_step,
                                            end_step=self.last_nn_outputs_saving_step,
                                            output_history=self.nn_output_history,
                                            output_sub_history=self.nn_output_sub_history,
                                            weight_history=weight_group,
                                            weight_sub_history=weight_sub_group})
                print('Saved:', data_filename)
                io.flush()
            end
            self.nn_output_history = {}
            self.nn_output_sub_history = {}
            -- self.nn_weight_history = {}
            -- self.nn_weight_sub_history = {}
            collectgarbage()

            -- create a new folder
            self.nn_outputs_dir_temp = self.nn_outputs_dir .. "/" .. step .. "-"
            lfs.mkdir(self.nn_outputs_dir_temp)
            self.dir_nn_outputs_step = step
        end

        -- save merged images
        image.save(self.nn_outputs_dir_temp.."/nn_data_".. self.num_offset+step ..".png", merged_img_output)

        -- update history data
        local ind = step - self.dir_nn_outputs_step + 1
        if output_group then
            self.nn_output_history[ind] = output_group
        end
        if output_sub_group then
            self.nn_output_sub_history[ind] = output_sub_group
        end
        -- if weight_group then
        --     self.nn_weight_history[ind] = weight_group
        -- end
        -- if weight_sub_group then
        --     self.nn_weight_sub_history[ind] = weight_sub_group
        -- end

        self.last_nn_outputs_saving_step = step
    end

    -- Show the screen when display available and display enabled
    if self.display then
        win_nn_output = image.display{image=merged_img_output, win=win_nn_output,min=0,max=1}
    end
end


--[[ Function
@description: get the history outputs of a specific layer at a certain step from a history data file
@input:
    history_file: the history data file
    step: the step of the data you want
    layer_num: the network layer of the data you want
@output: the data you want or "nil"
@notes:
]]
function visu:get_history_nn_outputs(history_file, step, layer_num)
    local err_msg, exp = pcall(torch.load, history_file)
    if not err_msg then
        error("Could not find the data file")
    end

    local start_step = exp.start_step
    local end_step = exp.end_step
    local step_limit = end_step - start_step + 1
    local history = exp.history
    local required_step = step - start_step + 1

    if required_step > step_limit or required_step < 1 then
        print("The required data are not in the provided data file.")
        return
    else
        local required_data = history[required_step][layer_num]
        print("The required data: (step "..step.."; layer "..layer_num..")\n",required_data)
        return required_data
    end
end