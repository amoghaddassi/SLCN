function maybe_switch_sides(trial)

global exp

% Determine when the last switch happened (if none yet, set to 1)
exp.last_switch_trial = find(exp.PROBSWITCHdata.switch_trial == 1, 1, 'last');
if isempty(exp.last_switch_trial)
    exp.last_switch_trial = exp.start_trial;
end

% Check if participant has gained the predetermind number of rewards for this run
number_rewards_in_run = nansum(exp.PROBSWITCHdata.reward(exp.last_switch_trial:trial-1));
if number_rewards_in_run >= exp.run_length(exp.run)
    exp.better_box_left = 1 - exp.better_box_left;  % Switch side of the better box
    exp.switch_trial = 1;   % Remember that this was a switch trial (reset to 0 in init_trial)
    exp.run = exp.run + 1;  % Keep track of which run we are currently in
    exp.debug_run_rewards = number_rewards_in_run;
    
end
