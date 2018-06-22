def fit_single_dataset(file_name):

    import numpy as np
    import os
    import pandas as pd
    from fit_parameters import FitParameters

    # Model parameters
    fit_par_names = ['alpha', 'beta', 'beta_high']
    learning_style = 'hierarchical'
    mix_probs = True

    # Which analyses should the script perform?
    run_on_cluster = False
    fit_model = True
    simulate_agents = False
    use_existing_data = True
    use_humans = True
    set_specific_parameters = False

    # Don't touch
    if run_on_cluster:
        base_path = '/home/bunge/maria/Desktop/Aliens'
        human_data_path = base_path + '/humanData/'
    else:
        base_path = 'C:/Users/maria/MEGAsync/Berkeley/TaskSets'
        human_data_path = base_path + '/Data/version3.1/'
    agent_data_path = base_path + '/AlienGenRec/'

    # How should the function be minimized?
    minimizer_stuff = {'save_plot_data': True,
                       'create_plot': not run_on_cluster,
                       'plot_save_path': agent_data_path,
                       'brute_Ns': 5,
                       'hoppin_T': 10.0,
                       'hoppin_stepsize': 0.5,
                       'NM_niter': 3,
                       'NM_xatol': .1,
                       'NM_fatol': .1,
                       'NM_maxfev': 10}

    # Task parameters
    n_actions = 3
                    # TS0
    TSs = np.array([[[1, 6, 1],  # alien0, items0-2
                     [1, 1, 4],  # alien1, items0-2
                     [5, 1, 1],  # etc.
                     [10, 1, 1]],
                    # TS1
                   [[1, 1, 2],  # alien0, items0-2
                    [1, 8, 1],  # etc.
                    [1, 1, 7],
                    [1, 3, 1]],
                    # TS2
                   [[1, 1, 7],  # TS2
                    [3, 1, 1],
                    [1, 3, 1],
                    [2, 1, 1]]])

    task_stuff = {'phases': ['1InitialLearning', '2CloudySeason', 'Refresher2', '3PickAliens',
                             'Refresher3', '5RainbowSeason', 'Mixed'],
                  'n_trials_per_alien': np.array([13, 10, 7, np.nan, 7, 1, 7]),  # np.array([1, 1, 1, np.nan, 1, 1, 1]), #
                  'n_blocks': np.array([3, 3, 2, np.nan, 2, 3, 3]),  # np.array([1, 1, 1, np.nan, 1, 1, 1]), #
                  'n_aliens': 4,
                  'n_actions': n_actions,
                  'n_contexts': 3,
                  'TS': TSs}

    comp_stuff = {'phases': ['season', 'alien-same-season', 'item', 'alien'],
                  'n_blocks': {'season': 3, 'alien-same-season': 3, 'item': 3, 'alien': 3}}

    agent_stuff = {'name': 'alien',
                   'n_TS': 3,
                   'beta_scaler': 2,
                   'beta_high_scaler': 10,
                   'learning_style': learning_style,
                   'mix_probs': mix_probs}

    parameters = {'fit_par_names': fit_par_names,
                  'par_names':
                  ['alpha', 'alpha_high', 'beta', 'beta_high', 'epsilon', 'forget', 'create_TS_biased_prefer_new', 'create_TS_biased_copy_old'],
                  'par_hard_limits':  # no values fitted outside; beta will be multiplied by 6 inside of alien_agents.py!
                  ((0., 1.), (0., 1.), (0., 1.), (0., 1.), (0., 1.),  (0., 1.), (0., 1.), (0., 1.)),
                  'par_soft_limits':  # no simulations outside
                  ((0., .5), (0., 1.), (0., 1.), (0., 1.), (0., .25), (0., .1), (0., 1.), (0., 1.)),
                  'default_pars':  # when a parameter is fixed
                  np.array([.1, 0.,    .1,       0.,       0.,        0.,       0.,       0.])}

    # Adjust things to user selection
    if use_existing_data:
        if use_humans:
            save_path = human_data_path
            file_name_pattern = 'aliens'
        else:
            save_path = agent_data_path
            file_name_pattern = 'sim_'
    else:
        save_path = agent_data_path
        file_name_pattern = 'sim_'

    parameters['fit_pars'] = np.array([par in parameters['fit_par_names'] for par in parameters['par_names']])
    fit_par_col_name = '_'.join([agent_stuff['learning_style'], '_'.join(parameters['fit_par_names'])])
    agent_stuff['fit_par'] = fit_par_col_name

    # Create folder to save output files
    if not os.path.isdir(save_path + '/fit_par/'):
        os.makedirs(save_path + '/fit_par/')

    # Create / fit each agent / person
    # Get agent id
    if use_existing_data:
        agent_id = int(file_name.split(file_name_pattern)[1].split('.csv')[0])
    else:
        agent_id = file_name
    print('\tPARTICIPANT {0}'.format(agent_id))
    agent_stuff['id'] = agent_id

    # Specify model
    fit_params = FitParameters(parameters=parameters, task_stuff=task_stuff,
                               comp_stuff=comp_stuff, agent_stuff=agent_stuff)

    # STEP 1: Get data
    if use_existing_data:
        print("Loading data {0}".format(file_name))
        agent_data = pd.read_csv(file_name)

    elif simulate_agents:
        gen_pars = np.array([lim[0] + np.random.rand() * (lim[1] - lim[0]) for lim in parameters['par_soft_limits']])
        gen_pars[np.invert(parameters['fit_pars'])] = 0
        if set_specific_parameters:
            for i, par in enumerate(gen_pars):
                change_par = input('Accept {0} of {1}? If not, type a new number.'.
                                   format(parameters['par_names'][i], np.round(par, 2)))
                if change_par:
                    gen_pars[i] = float(change_par)
        print('Simulating {0} agent {1} with parameters {2}'.format(
            agent_stuff['learning_style'], agent_id, np.round(gen_pars, 3)))
        agent_data = fit_params.simulate_agent(all_pars=gen_pars)

        created_file_name = save_path + file_name_pattern + str(agent_id) + ".csv"
        print("Saving simulated data to {0}".format(created_file_name))
        agent_data.to_csv(created_file_name)
    agent_data['sID'] = agent_id  # redundant but crashes otherwise...

    # STEP 2: Fit parameters
    if fit_model:

        # Clean data
        if use_humans:
            # Remove all rows that do not contain 1InitialLearning data (-> jsPysch format)
            agent_data = agent_data.rename(columns={'TS': 'context'})  # rename "TS" column to "context"
            context_names = [str(TS) for TS in range(agent_stuff['n_TS'])]
            item_names = range(task_stuff['n_actions'])
            agent_data = agent_data.loc[
                (agent_data['context'].isin(context_names)) & (agent_data['item_chosen'].isin(item_names))]
        else:
            # Remove all phases that are not InitialLearning
            agent_data = agent_data.loc[agent_data['phase'] == '1InitialLearning']
        agent_data.index = range(agent_data.shape[0])

        if 'alpha' in agent_data.columns:
            # Look up generated parameters and display
            gen_pars = agent_data.loc[0, parameters['par_names']]
            gen_pars[2:4] /= agent_stuff['beta_scaler']  # beta
            print("True parameters: {0}".format(np.round(gen_pars.values.astype(np.double), 3)))

        # Find parameters that minimize NLL
        rec_pars = fit_params.get_optimal_pars(agent_data=agent_data, minimizer_stuff=minimizer_stuff)
        print("Fitted parameters: {0}".format(np.round(rec_pars, 3)))

        # Calculate NLL,... of minimizing parameters
        agent_data = fit_params.calculate_NLL(vary_pars=rec_pars[np.argwhere(parameters['fit_pars'])],
                                              agent_data=agent_data,
                                              goal='add_decisions_and_fit',
                                              suff='_rec')

    # Write agent_data as csv
    created_file_name = save_path + '/fit_par/' + file_name_pattern + str(agent_id) + ".csv"
    print("Saving fitted data to {0}".format(created_file_name))
    agent_data.to_csv(created_file_name)