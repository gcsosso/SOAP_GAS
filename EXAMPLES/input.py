# Input parameters for gene sets, read docs for information on what they do
descDict1 = {'lower': 2,
             'upper': 10,
             'centres': '{8, 7, 6, 1, 16, 17, 9}',
             'neighbours': '{8, 7, 6, 1, 16, 17, 9}',
             'nu_R': 1,
             'nu_S': 0,
             'mutation_chance': 0.50,
             'min_cutoff': 5,
             'max_cutoff': 20,
             'min_sigma': 0.1,
             'max_sigma': 1.5,
             'message_steps': 0}


population_parameters = {'best_sample': 6,
                         'lucky_few': 2,
                         'population_size': 20,
                         'number_of_children': 5,
                         'maximise_scores': False}


history_parameters = {'early_stop': 0.03,
                      'early_number': 3,
                      'min_generations': 5}

# List of parameters, you might want multiple GeneSets if you are doing things
# like double soaps.
descList = [descDict1]
num_gens = 20
