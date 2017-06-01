function pr_choice(trial)

global exp
clearkeys;

%% Check if participant chose a valid key and highlight choice if so
trial_fly = exp.butterfly_sequence(trial);
if ~isempty(exp.key)
    exp.key = exp.key(1);
    if exp.key == exp.nkey.le
        if exp.show_points
            preparestring(strcat('Points: ', num2str(exp.earned_points)), exp.buffer.le_choice, exp.p.counter_x,300);
        end
        drawpict(exp.buffer.le_choice);
        wait(exp.times.display_choice);
        % if that was the correct choice, mark that
        if sum(trial_fly == exp.le_flower_butterflies)
            exp.ACC = 1;
            exp.correct_response = 1;
        else
            exp.ACC = 0;
            exp.correct_response = 0;
        end

    % When participant picked the right flower, highlight the right flower
    elseif exp.key == exp.nkey.ri
        if exp.show_points
            preparestring(strcat('Points: ', num2str(exp.earned_points)), exp.buffer.ri_choice, exp.p.counter_x,300);
        end
        drawpict(exp.buffer.ri_choice);
        wait(exp.times.display_choice);
        % if that was the correct choice, mark that
        if sum(trial_fly == exp.ri_flower_butterflies)
            exp.ACC = 1;
            exp.correct_response = 1;
        else
            exp.ACC = 0;
            exp.correct_response = 0;
        end
    % When participant pressed a wrong key
    else
        if exp.show_points
            preparestring(strcat('Points: ', num2str(exp.earned_points)), exp.buffer.wrong_key,exp.p.counter_x300);
        end
        drawpict(exp.buffer.wrong_key);
        wait(2 * exp.times.display_choice);
        exp.ACC = nan;
    end
% When participants gave no response
else
    if exp.show_points
        preparestring(strcat('Points: ', num2str(exp.earned_points)), exp.buffer.response_reminder,exp.p.counter_x,300);
        preparestring(strcat('Points: ', num2str(exp.earned_points)), exp.buffer.no_response,exp.p.counter_x,300);
    end
    drawpict(exp.buffer.no_response);
    wait(2 * exp.times.display_choice);
    exp.ACC = nan;
end

% Count number of correct and incorrect responses (to move on in reward_sequence vector)
if exp.ACC == 1  % correct response
    exp.n_correct(trial_fly) = exp.n_correct(trial_fly) + 1;
elseif exp.ACC == 0  % incorrect response
    exp.n_incorrect(trial_fly) = exp.n_incorrect(trial_fly) + 1;
end  % no response