--[[ File
@description:
    This class is for evaluating robotic reaching performance, with a focus on reinforcement learning cases.
@version: V0.15
@author: Fangyi Zhang   email:gzzhangfangyi@gmail.com
@acknowledgement:
    ARC Centre of Excellence for Robotic Vision (ACRV)
    Queensland Univsersity of Technology (QUT)
@history:
    V0.00   06/08/2015  developed the first version
    V0.10   13/08/2015  added the performance assessment function
    V0.11   20/11/2015  updated performance assessment function: controlled by the input state
    V0.12   28/11/2015  fixed a bug that self.nepisodes might be zero as a divider in the function of performance_assessment
    V0.13   10/12/2015  added real Q value assessment
    V0.15   27/07/2018  added some comments for code re-organization

]]

require "torch"
require "common/curve_plot"
require "data_analysis/data_processing"

-- construct a class
evalu = torch.class('evaluation')

--[[ Function
@description: initialize an evaluation object
@input:
    args: settings for an evaluation object, i.e., {agent=agent, options=opt}
@output: nil
@notes:
]]
function evalu:__init(args)

    self.agent = args.agent
    self.eval_steps = args.options.eval_steps or 1000
    self.plot_filename = args.options.plot_filename..".png"
    self.efficiency_filename = args.options.plot_filename.."_efficiency.png"
    self.success_rate_filename = args.options.plot_filename.."_success_rate.png"
    self.Q_value_filename = args.options.plot_filename.."_real_Q_value.png"

    self.total_reward = 0
    self.nrewards = 0
    self.nepisodes = 0
    self.episode_reward = 0
    self.eval_time = sys.clock()
    self.start_time = sys.clock()

    self.reward_counts = {}
    self.episode_counts = {}
    self.v_history = {}
    self.qmax_history = {}
    self.td_history = {}
    self.reward_history = {}
    self.time_history = {}
    self.time_history[1] = 0

    -- Defined for efficiency and success rate assessment
    self.efficiency = 0
    self.nepisode_reward = 0
    self.efficiency_history = {} -- task completion efficiency, i.e., reward/step_cost
    self.aver_Q = {} -- average Q value in the test, i.e., total_reward/nepisodes
    self.ncompletion = 0
    self.success_rate_history = {} -- task completion rate
    self.episode_reward_history = {} -- episode reward
    self.episode_closest_distance_history = {}
    self.episode_final_distance_history = {}
    self.episode_initial_distance_history = {}
    self.episode_closest_completion_ratio_history = {}
    self.episode_final_completion_ratio_history = {}

    -- Initialize a curve plotting object
    self.curve_plt = data_processing{}
    self.network_name = args.options.network

    -- Define a window when display enabled and display available
    if (args.options.display_avail == 1) then
        if (args.options.display == 1) then
            self.display = true
        end
    end
end

--TODO: develop better evaluation functions, what to evaluate
--[[ Function
@description: evaluate the performance
@input:
    step: the current testing step number
    reward: the current reward value
    terminal: the current terminal value
@output: nil
@notes: have not yet found this function being used in any scripts with its current version, to double check
]]
function evalu:evaluation(step, reward, terminal)
    -- reset counting variables at the beginning of a new evaluation period
    if step%self.eval_steps == 1 then
        self.total_reward = 0
        self.nrewards = 0
        self.nepisodes = 0
        self.episode_reward = 0
        self.eval_time = sys.clock()
    end

    -- record every reward
    self.episode_reward = self.episode_reward + reward
    if reward ~= 0 then
       self.nrewards = self.nrewards + 1
    end

    if terminal then
        self.total_reward = self.total_reward + self.episode_reward
        self.episode_reward = 0
        self.nepisodes = self.nepisodes + 1
    end

    -- calculate the evaluation results
    if step%self.eval_steps == 0 then
        self.eval_time = sys.clock() - self.eval_time
        self.start_time = self.start_time + self.eval_time
        self.agent:compute_validation_statistics() -- Compute Validation Statistics
        local ind = #self.reward_history+1
        self.total_reward = self.total_reward/math.max(1, self.nepisodes)

        if self.agent.v_avg then
            self.v_history[ind] = self.agent.v_avg
            self.td_history[ind] = self.agent.tderr_avg
            self.qmax_history[ind] = self.agent.q_max
        end
        print("V", self.v_history[ind], "TD error", self.td_history[ind], "Qmax", self.qmax_history[ind])

        -- plot average estimated Q value
        local v_hist_tensor = torch.Tensor(self.v_history)

        if self.display then
            plotTensorToDisplay(v_hist_tensor, 'v_history', 'Training Epochs', 'Average Action Value (Q)', 1)
        end
        plotTensorToFile(v_hist_tensor, 'v_history', 'Training Epochs', 'Average Action Value (Q)', self.plot_filename)

        -- update history data
        self.reward_history[ind] = self.total_reward
        self.reward_counts[ind] = self.nrewards
        self.episode_counts[ind] = self.nepisodes

        self.time_history[ind+1] = sys.clock() - self.start_time

        local time_dif = self.time_history[ind+1] - self.time_history[ind]

        print(string.format(
            '\nSteps: %d, reward: %.2f, epsilon: %.2f, lr: %G, ' ..
            'testing time: %ds, num. ep.: %d,  num. rewards: %d',
            step, self.total_reward, self.agent.ep, self.agent.lr, time_dif,
            self.nepisodes, self.nrewards))
    end
end


--[[ Function
@description: evaluate the efficiency and success rate
@input:
    reward: the current reward value
    terminal: the current terminal value
    completion: the current completion value
    state: performance assessment state: "initialize", "counting", "end"
@output: nil
@notes: this function is now used most widely in the framework with its current version, for reaching performance evaluation
]]
function evalu:performance_assessment(reward, terminal, completion, state)

    if state == "counting" then
        -- record every reward
        self.episode_reward = self.episode_reward + reward
        self.nrewards = self.nrewards + 1
        self.nepisode_reward = self.nepisode_reward + 1

        if terminal then
            if completion then
                self.ncompletion = self.ncompletion + 1
            end
            self.total_reward = self.total_reward + self.episode_reward
            self.efficiency = self.efficiency + self.episode_reward / self.nepisode_reward
            self.episode_reward = 0
            self.nepisode_reward = 0
            self.nepisodes = self.nepisodes + 1
        end

    -- reset counting variables at the beginning of a new evaluation period
    elseif state == "initialize" then
        self.total_reward = 0
        self.nepisodes = 0
        self.ncompletion = 0
        self.episode_reward = 0
        self.nepisode_reward = 0
        self.efficiency = 0

    -- calculate the evaluation results
    elseif state == "end" then
        local ind = #self.reward_history+1

        -- update history data
        if self.nepisodes > 0 then
            self.efficiency_history[ind] = self.efficiency / self.nepisodes
            self.success_rate_history[ind] = self.ncompletion / self.nepisodes
            self.aver_Q[ind] = self.total_reward / self.nepisodes
        else
            self.efficiency_history[ind] = 0
            self.success_rate_history[ind] = 0
            self.aver_Q[ind] = 0
        end
        self.reward_history[ind] = self.total_reward
        self.episode_counts[ind] = self.nepisodes

        print("============================================================")
        print("Average Q Value:", self.aver_Q[ind], "Efficiency:", self.efficiency_history[ind], "Success Rate:", self.success_rate_history[ind], "nepisodes:", self.episode_counts[ind])

        -- plot average efficiency and success rate
        local efficiency_history_tensor = torch.Tensor(self.efficiency_history)
        local success_rate_history_tensor = torch.Tensor(self.success_rate_history)
        local average_Q_value_tensor = torch.Tensor(self.aver_Q)
        if self.display then
            plotTensorToDisplay(efficiency_history_tensor, 'Efficiency History', 'Training Epochs', 'Average Efficiency (reward / nrewards in each episode)', 2)
            plotTensorToDisplay(success_rate_history_tensor, 'Success Rate History', 'Training Epochs', 'Success Rate (ncompletion / nepisodes)', 3)
            plotTensorToDisplay(average_Q_value_tensor, 'Average Q Value History', 'Training Epochs', 'Average Q Value (total_reward / nepisodes)', 4)
        end
        plotTensorToFile(efficiency_history_tensor, 'Efficiency History', 'Training Epochs', 'Average Efficiency (reward / nrewards in each episode)', self.efficiency_filename)
        plotTensorToFile(success_rate_history_tensor, 'Success Rate History', 'Training Epochs', 'Success Rate (ncompletion / nepisodes)', self.success_rate_filename)
        plotTensorToFile(average_Q_value_tensor, 'Average Q Value History', 'Training Epochs', 'Average Q Value (total_reward / nepisodes)', self.Q_value_filename)

    end
end


--[[ Function
@description: evaluate the efficiency and success rate for ACRA 2015
@input:
    step: the current testing step number
    reward: the current reward value
    terminal: the current terminal value
    completion: the current completion value
@output: nil
@notes: this function is not used in the latest framework, just leave here for history record
]]
function evalu:performance_assessment_ACRA(step, reward, terminal, completion)
    -- reset counting variables at the beginning of a new evaluation period
    if step == 1 then
        self.nepisodes = 0
        self.ncompletion = 0
        self.episode_reward = 0
    end

    -- record every reward
    self.episode_reward = self.episode_reward + reward
    -- print("Episode_Reward:", self.episode_reward)

    if terminal then
        if completion then
            self.ncompletion = self.ncompletion + 1
        end
        local ind = #self.episode_reward_history+1
        -- print("Episode_Reward:", self.episode_reward)
        self.episode_reward_history[ind] = self.episode_reward
        self.episode_reward = 0
        self.nepisodes = self.nepisodes + 1
        -- print("n_Episode:", self.nepisodes)
    end

    -- calculate the evaluation results
    if self.nepisodes==self.eval_steps then
        local success_rate_temp = self.ncompletion / self.nepisodes
        local reward_tensor = torch.Tensor(self.episode_reward_history)
        local reward_mean = torch.mean(reward_tensor)
        local reward_variance = torch.var(reward_tensor)


        self.curve_plt:saveMeanToTxt(reward_mean, reward_variance, success_rate_temp, "testing_data"..self.network_name..".txt")

        -- save data used for figuring
        torch.save("testing_data"..self.network_name..".t7", {reward_history = self.episode_reward_history,
                                        success_rate = success_rate_temp,
                                        reward_mean = reward_mean,
                                        reward_variance = reward_variance,
                                        n_episodes = self.nepisodes,
                                        n_completion = self.ncompletion})
        print('Saved:', "testing_data"..self.network_name..".t7")
        print("Reward_mean:", reward_mean, "Reward_variance:", reward_variance, "Success Rate:", success_rate_temp, "nepisodes:", self.nepisodes)
    end
end


--[[ Function
@description: evaluate the distance statistic results for ICRA 2017 and ACRA 2017
@input:
    step: the current testing step number
    reward: the current reward value
    terminal: the current terminal value
    completion: the current completion value
@output: nil
@notes: this function is not used in the latest framework, just leave here for history record
]]
function evalu:performance_assessment_ICRA(step, reward, terminal, completion, initial_distance, distance, closest_distance)
    -- reset counting variables at the beginning of a new evaluation period
    if step == 1 then
        self.nepisodes = 0
        self.ncompletion = 0
        self.episode_reward = 0
    end

    -- record every reward
    self.episode_reward = self.episode_reward + reward
    -- print("Episode_Reward:", self.episode_reward)

    if terminal then
        if completion then
            self.ncompletion = self.ncompletion + 1
        end
        local ind = #self.episode_reward_history+1
        -- print("Episode_Reward:", self.episode_reward)
        self.episode_reward_history[ind] = self.episode_reward
        self.episode_closest_distance_history[ind] = closest_distance
        self.episode_final_distance_history[ind] = distance
        self.episode_initial_distance_history[ind] = initial_distance
        self.episode_closest_completion_ratio_history[ind] = 1 - closest_distance / initial_distance
        self.episode_final_completion_ratio_history[ind] = 1 - distance / initial_distance
        self.episode_reward = 0
        self.nepisodes = self.nepisodes + 1
        -- print("n_Episode:", self.nepisodes)
    -- end

        -- calculate the evaluation results
        if self.nepisodes > 0 and self.nepisodes <= self.eval_steps then
            local success_rate_temp = self.ncompletion / self.nepisodes
            local reward_tensor = torch.Tensor(self.episode_reward_history)
            local reward_mean = torch.mean(reward_tensor)
            local reward_variance = torch.var(reward_tensor)
            self.curve_plt:saveMeanToTxt(reward_mean, reward_variance, success_rate_temp, "reward_success_rate"..self.network_name..".txt")

            local closest_dis_tensor = torch.Tensor(self.episode_closest_distance_history)
            local closest_dis_mean = torch.mean(closest_dis_tensor)
            local closest_dis_variance = torch.var(closest_dis_tensor)
            self.curve_plt:saveStatisticToTxt(closest_dis_mean, closest_dis_variance, "closest_dis", "closest_dis"..self.network_name..".txt")

            local final_dis_tensor = torch.Tensor(self.episode_final_distance_history)
            local final_dis_mean = torch.mean(final_dis_tensor)
            local final_dis_variance = torch.var(final_dis_tensor)
            self.curve_plt:saveStatisticToTxt(final_dis_mean, final_dis_variance, "final_dis", "final_dis"..self.network_name..".txt")

            local initial_dis_tensor = torch.Tensor(self.episode_initial_distance_history)
            local initial_dis_mean = torch.mean(initial_dis_tensor)
            local initial_dis_variance = torch.var(initial_dis_tensor)
            self.curve_plt:saveStatisticToTxt(initial_dis_mean, initial_dis_variance, "initial_dis", "initial_dis"..self.network_name..".txt")

            local closest_comp_tensor = torch.Tensor(self.episode_closest_completion_ratio_history)
            local closest_comp_mean = torch.mean(closest_comp_tensor)
            local closest_comp_variance = torch.var(closest_comp_tensor)
            self.curve_plt:saveStatisticToTxt(closest_comp_mean, closest_comp_variance, "closest_comp", "closest_comp"..self.network_name..".txt")

            local final_comp_tensor = torch.Tensor(self.episode_final_completion_ratio_history)
            local final_comp_mean = torch.mean(final_comp_tensor)
            local final_comp_variance = torch.var(final_comp_tensor)
            self.curve_plt:saveStatisticToTxt(final_comp_mean, final_comp_variance, "final_comp", "final_comp"..self.network_name..".txt")

            -- save data used for figuring
            torch.save("testing_data"..self.network_name..".t7", {reward_history = self.episode_reward_history,
                                            closest_distance_history = self.episode_closest_distance_history,
                                            final_distance_history = self.episode_final_distance_history,
                                            initial_distance_history = self.episode_initial_distance_history,
                                            closest_completion_ratio_history = self.episode_closest_completion_ratio_history,
                                            final_completion_ratio_history = self.episode_final_completion_ratio_history,
                                            success_rate = success_rate_temp,
                                            reward_mean = reward_mean,
                                            reward_variance = reward_variance,
                                            n_episodes = self.nepisodes,
                                            n_completion = self.ncompletion})
            print('Saved:', "testing_data"..self.network_name..".t7")
            print("Reward_mean:", reward_mean, "Reward_variance:", reward_variance, "Success Rate:", success_rate_temp, "nepisodes:", self.nepisodes)
        end
    end
end
