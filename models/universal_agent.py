import numpy as np


class UniversalAgent(object):
    def __init__(self, agent_stuff, all_params_lim, task_stuff):
        self.n_actions = task_stuff['n_actions']
        self.learning_style = agent_stuff['learning_style']
        self.id = agent_stuff['id']
        self.method = agent_stuff['method']
        [self.alpha, self.beta, self.epsilon, self.perseverance, self.decay] = all_params_lim
        if self.learning_style == 'RL':
            self.initial_value = 1 / self.n_actions
            self.q = self.initial_value * np.ones(self.n_actions)
        elif self.learning_style == 'Bayes':
            self.p_switch = task_stuff['p_reward'] / np.mean(task_stuff['av_run_length'])  # true average switch probability
            self.p_reward = task_stuff['p_reward']  # true reward probability
            self.p_boxes = np.ones(self.n_actions) / self.n_actions  # initialize prior uniformly over all actions
            self.initial_value = self.p_boxes.copy()
        self.previous_action = np.zeros(self.n_actions)
        self.LL = 0
        self.p_actions = np.ones(self.n_actions) / self.n_actions

    # Take action
    def calculate_p_actions(self):
        action_values = self._get_action_values()
        sticky_values = action_values + self.perseverance * self.previous_action
        sticky_values[sticky_values <= 0.001] = 0.001
        if self.method == 'epsilon-greedy':
            if sticky_values[0] == sticky_values[1]:
                self.p_actions = np.ones(self.n_actions) / self.n_actions
            else:
                self.p_actions = self.epsilon * np.ones(self.n_actions)
                better_action = np.argwhere(sticky_values == np.nanmax(sticky_values))[0]
                self.p_actions[better_action] = 1 - self.epsilon
        elif self.method == 'softmax':
            p_left_box = 1 / (1 + np.exp(self.beta * (sticky_values[1] - sticky_values[0])))
            self.p_actions = np.array([p_left_box, 1 - p_left_box])
        self.p_actions = 0.999 * self.p_actions + 0.001 / self.n_actions  # avoid 0's & 1's

    def select_action(self):
        return int(np.random.rand() > self.p_actions[0])  # select left ('0') when np.random.rand() < p_actions[0]

    def _get_action_values(self):
        if self.learning_style == 'RL':
            return self.q
        else:
            return self.p_boxes

    # Learn
    def learn(self, action, reward):
        self.previous_action = np.array(range(self.n_actions)) == action
        self._update_LL(action)
        if self.learning_style == 'RL':
            self._update_q(action, reward)
        elif self.learning_style == 'Bayes':
            self._update_p_boxes(action, reward)

    # Learn helpers
    def _update_LL(self, action):
        self.LL += np.log(self.p_actions[action])

    def _update_p_boxes(self, action, reward):
        self.p_boxes += self.decay * (self.initial_value - self.p_boxes)  # decay values back to uniform
        if reward == 1:  # The probability of getting a reward is 0 for all actions except the correct one
            lik_boxes = np.zeros(self.n_actions) + 0.001  # avoid likelihoods of 0
            lik_boxes[action] = self.p_reward
        else:  # The probability of getting NO reward is 1 for all actions except the correct one
            lik_boxes = np.ones(self.n_actions)
            lik_boxes[action] = 1 - self.p_reward
        lik_times_prior = lik_boxes * self.p_boxes  # "*" -> element-wise multiplication
        posterior = lik_times_prior / np.sum(lik_times_prior)  # normalize such that sum == 1
        next_trial_if_no_switch = (1 - self.p_switch) * posterior
        next_trial_if_switch = self.p_switch * np.flipud(posterior)
        self.p_boxes = next_trial_if_no_switch + next_trial_if_switch

    def _update_q(self, action, reward):
        self.q += self.decay * (self.initial_value - self.q)  # decay values back to initial value
        self.q[action] += self.alpha * (reward - self.q[action])  # update value of chosen action
