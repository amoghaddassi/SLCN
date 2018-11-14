import numpy as np
import pandas as pd
import glob
import os
import re


class Task(object):

    def __init__(self, n_subj, n_seasons):

        self.n_subj = n_subj
        self.n_seasons = n_seasons
        #Need to change size of self.TS with the n_seasons parameter
        self.TS = np.array([[[1, 6, 1],  # alien0, items0-2
                             [1, 1, 4],  # alien1, items0-2
                             [5, 1, 1],  # etc.
                             [10, 1, 1]],
                            # TS1
                            [[1, 1, 2],  # alien0, items0-2
                             [1, 8, 1],  # etc.
                             [1, 1, 7],
                             [1, 3, 1]],
                            # TS2
                            [[1, 1, 7],
                             [3, 1, 1],
                             [1, 3, 1],
                             [2, 1, 1]],
                            [[1, 6, 1],  # alien0, items0-2
                             [1, 1, 4],  # alien1, items0-2
                             [5, 1, 1],  # etc.
                             [10, 1, 1]],
                            # TS1
                            [[1, 1, 2],  # alien0, items0-2
                             [1, 8, 1],  # etc.
                             [1, 1, 7],
                             [1, 3, 1]],
                            # TS2
                            [[1, 1, 7],
                             [3, 1, 1],
                             [1, 3, 1],
                             [2, 1, 1]]])

    def get_trial_sequence(self, file_path, n_subj, n_sim_per_subj, subset_of_subj, fake=False,
                           phases=("1InitialLearning", "Refresher2", "Refresher3")):
        '''
        Get trial sequences of human participants.
        Read in datafiles of all participants, select InitialLearning, Refresher2, and Refresher3,
        and get seasons and aliens in each trial.
        :param file_path: path to human datafiles
        :param n_subj: number of files to be read in
        :param n_sim_per_subj: number of simulations per subject (duplicates season and alien sequences)
        :param fake: create season & alien sequences rather than using humans' (default: False, i.e., use humans')
        :param phases: data from which task phases should be used when reading in humans' season & alien sequences?
        (defaults to ("1InitialLearning", "Refresher2", "Refresher3"))
        :return: n_trials: number of trials in each file
        '''

        filenames = glob.glob(os.path.join(file_path, '*.csv'))
        n_trials = 100000
        #for filename in np.array(filenames)[subset_of_subj][:n_subj]:
        for filename in filenames:
            agent_data = pd.read_csv(filename)

            # Remove all rows that do not contain 1InitialLearning data (-> jsPysch format)
            TS_names = [str(TS) for TS in range(3)]
            agent_data = agent_data.loc[(agent_data['TS'].isin(TS_names)) &
                                        (agent_data['phase'].isin(phases))]

            # Read out sequence of seasons and aliens
            try:
                seasons = np.hstack([seasons, agent_data["TS"]])
                aliens = np.hstack([aliens, agent_data["sad_alien"]])
                phase = np.hstack([phase, agent_data["phase"]])
                correct = np.hstack([correct, agent_data["correct"]])
            except NameError:
                seasons = agent_data["TS"]
                aliens = agent_data["sad_alien"]
                phase = agent_data["phase"]
                correct = agent_data["correct"]
            n_trials = np.min([n_trials, agent_data.shape[0]])

        # Bring into right shape
        seasons = np.tile(seasons, n_sim_per_subj)
        aliens = np.tile(aliens, n_sim_per_subj)
        self.seasons = seasons.reshape([int(len(seasons) / n_trials), n_trials]).astype(int).T
        self.aliens = aliens.reshape([int(len(aliens) / n_trials), n_trials]).astype(int).T
        self.phase = np.tile(agent_data["phase"], n_sim_per_subj)

        if fake:
            #TODO: adjust season sequences so the same TS doesn't appear twice in a row
            self.ts_orders = [[0, 1, 2, 3, 4, 5],
                                [1, 2, 0, 4, 5, 3],
                                [2, 0, 1, 5, 3, 4]]
            self.seasons = np.zeros([self.n_subj, self.n_seasons*80*4])
            for i in range(self.n_subj):
                order = np.random.choice(range(3))
                sequence = np.tile(np.repeat(self.ts_orders[order], 80), 4)
                self.seasons[i, :] = sequence
            self.seasons = self.seasons.T.astype(np.int64)  # np.zeros([n_subj, n_trials], dtype=int).T
            print(self.seasons)
            self.aliens = np.tile(np.arange(4), int(n_subj * 80 * 6)).reshape([n_subj, self.n_seasons * 80 * 4]).astype(int).T
            n_trials = self.seasons.shape[0]
            self.phase = '1InitialLearning'

        #return n_trials, correct.reshape([int(len(correct) / n_trials), n_trials]).astype(int).T
        return n_trials, correct

    def present_stimulus(self, trial):

        # Look up alien and context for current trial
        self.alien = self.aliens[trial]  # , :self.n_subj * ]
        self.season = self.seasons[trial]  # , :self.n_subj]

        return self.season, self.alien

    def produce_reward(self, action):

        # Look up reward in TS table, determine if response was correct
        reward = self.TS[self.season, self.alien, action]
        correct = reward > 1

        # Add noise to reward
        noised_reward = np.round(np.random.normal(reward, 0.05), 2)
        noised_reward[noised_reward < 0] = 0

        return noised_reward, correct
