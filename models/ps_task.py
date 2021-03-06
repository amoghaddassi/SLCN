import numpy as np
import scipy.io as spio


class Task(object):
    def __init__(self, task_stuff=None, agent_id=0):
        self.n_actions = task_stuff['n_actions']
        self.p_reward = task_stuff['p_reward']
        self.n_trials = task_stuff['n_trials']
        self.path = task_stuff['path']
        self.correct_box = int(np.random.rand() > 0.5)  # 0 or 1
        self.n_rewards = 0
        self.n_correct = 0
        self.i_episode = 0
        self.switched = False
        self.reward_version = str(agent_id % 4)
        self.run_length = spio.loadmat(self.path + '/run_length' + self.reward_version + '.mat',
                                       squeeze_me=True)['run_length']
        self.coin_win = spio.loadmat(self.path + '/coin_win' + self.reward_version + '.mat',
                                     squeeze_me=True)['coin_win']

    def prepare_trial(self):

        # switch box if necessary
        period_over = self.n_rewards == self.run_length[self.i_episode]
        if period_over:
            self.correct_box = 1 - self.correct_box
            self.switched = True
            self.n_rewards = 0
            self.i_episode += 1

    def produce_reward(self, action):

        # Determine whether agent's action was correct
        correct = action == self.correct_box

        # Determine whether a reward should be given
        if correct:
            reward = self.coin_win[self.n_correct]  # get predetermined reward
            if reward == 0 and self.switched:  # the first trial after a switch is always rewarded
                reward = 1
                self._exchange_rewards()
                self.switched = False

            # Keep count of things (can I move this outside the loop?)
            self.n_correct += 1
            self.n_rewards += reward
        else:
            reward = 0
        return reward, correct

    def _exchange_rewards(self):

        # Exchange rewards if necessary
        reward_indexes = np.argwhere(self.coin_win == 1)
        next_reward_index = reward_indexes[reward_indexes > self.n_correct][0]
        self.coin_win[next_reward_index] = 0
