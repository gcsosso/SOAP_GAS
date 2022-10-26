import pandas as pd
from quippy import descriptors
import ase.io
import subprocess
from pathlib import Path
import os
import random
import pickle as pkl

import numpy as np


class GeneSet:
    def __init__(self, cutoff: int, l_max: int, n_max: int, sigma: int, **kwargs) -> None:
        self.cutoff = cutoff
        self.l_max = l_max
        self.n_max = n_max
        self.sigma = sigma
        allowed_keys = set(['centres', 'neighbours', 'mu', 'mu_hat', 'nu', 'nu_hat'])
        # update the given keys by their given values
        self.__dict__.update((key, value) for key, value in kwargs.items() if key in allowed_keys)


class Individual:
    def __init__(self, gene_set_list):
        self.gene_set_list = gene_set_list
        parameter_string_list = [get_parameter_string(gene_set) for gene_set in gene_set_list]
        soaps, targets = comp_soaps(parameter_string_list, conf_s, target_dataframe)
        self.score = comp_score(soaps, targets)

    def __le__(self, other):
        return self.score <= other.score

    def __ge__(self, other):
        return self.score >= other.score

    def __lt__(self, other):
        return self.score < other.score

    def __gt__(self, other):
        return self.score > other.score


def comp_score(soaps, targets):
    return random.randint(1, 100)


def comp_random_genes(lower: int,
                      upper: int,
                      min_cutoff: int,
                      max_cutoff: int,
                      min_sigma: float,
                      max_sigma: float,
                      **kwargs) -> GeneSet:
    """
    This takes the parameters given in the input file and returns an instance of
    an GeneSet class with random parameters.
    """
    gene_parameters = kwargs
    gene_parameters['cutoff'] = np.random.randint(min_cutoff, max_cutoff)
    gene_parameters['l_max'] = np.random.randint(lower, upper)
    gene_parameters['n_max'] = np.random.randint(lower, upper)
    gene_parameters['sigma'] = round(np.random.uniform(min_sigma, max_sigma), 2)
    return GeneSet(**gene_parameters)


def initialise_gene_set(desc_dict_list: list[dict], pop_size: int) -> list:
    """
    Creates the initial population of size popSize by generating a GeneSet
    for each set of parameters in the input file
    """
    return [[comp_random_genes(**desc_dict) for desc_dict in desc_dict_list] for _ in range(pop_size)]


def get_population(list_of_genesets: list[list[GeneSet]]) -> list[Individual]:
    """
    This function creates a population of individuals based on the genes given as input
    """
    return [Individual(gene_set_list) for gene_set_list in list_of_genesets]


def comp_soaps(parameter_string_list, conf_s, data):
    print(parameter_string_list)
    soap_array = []
    targets = np.array(data["Target"])
    for mol in conf_s:
        soap = []
        for parameter_string in parameter_string_list:
            soap += list(descriptors.Descriptor(parameter_string).calc(mol)['data'])
        soap_array.append(soap)
    return np.array(soap_array), np.array(targets)


def get_parameter_string(gene_set: GeneSet) -> str:
    """
    Creates the string needed to generate SOAP descriptor
    """
    num_centre_atoms = gene_set.centres.count(',') + 1
    num_neighbour_atoms = gene_set.neighbours.count(',') + 1
    return "soap average cutoff={cutoff} l_max={l_max} n_max={n_max} atom_sigma={sigma} n_Z={0} Z={centres} " \
           "n_species={1} species_Z={neighbours} mu={mu} mu_hat={mu_hat} nu={nu} nu_hat={nu_hat}".format(
        num_centre_atoms, num_neighbour_atoms, **vars(gene_set))


def get_conf():
    path = str(os.path.dirname(os.path.abspath(__file__))) + "/EXAMPLES/"
    if os.path.isfile(path + "conf_s.pkl"):
        print("conf_s file exists")
        return [*pkl.load(open(path + "conf_s.pkl", "rb"))]
    else:
        try:
            csv_path = Path(path + "database.csv").resolve(strict=True)
            xyz_folder_path = Path(path + "xyz/").resolve(strict=True)
        except FileExistsError:
            print("Please make sure the xyz folder and database.csv file exist. Read the docs for more information.")
        print("Generating conf")
        names_and_targets = pd.read_csv(csv_path)
        conf_s = []
        for index, row in names_and_targets.iterrows():
            xyzname = str(xyz_folder_path) + '/' + row['Name'] + '.xyz'
            subprocess.call("sed 's/CL/Cl/g' " + xyzname + " | sed 's/LP/X/g' > tmp.xyz", shell=True)
            conf = ase.io.read("tmp.xyz")
            conf_s.append(conf)
        subprocess.call("rm tmp.xyz", shell=True)
        print(f"saving conf to {path}conf_s.pkl")
        pkl.dump([conf_s, names_and_targets], open(path + 'conf_s.pkl', 'wb'))
        return [conf_s, names_and_targets]

def next_generation(previous_generation: list[Individual], best_sample: int, lucky_few: int, number_of_children: int) -> list[Individual]:
    pass


def Main():
    descDict1 = {'lower': 2, 'upper': 6, 'centres': '{8, 7, 6, 1, 16, 17, 9}', 'neighbours': '{8, 7, 6, 1, 16, 17, 9}',
                 'mu': 0, 'mu_hat': 0, 'nu': 2, 'nu_hat': 0, 'average': True, 'mutationChance': 0.15, 'min_cutoff': 5,
                 'max_cutoff': 10, 'min_sigma': 0.1, 'max_sigma': 0.5}
    descDict2 = {'lower': 2, 'upper': 6, 'centres': '{8, 7, 6, 1, 16, 17, 9}', 'neighbours': '{8, 7, 6, 1, 16, 17, 9}', 'mu': 0,
                 'mu_hat': 0, 'nu': 2, 'nu_hat': 0, 'average': True, 'mutationChance': 0.15, 'min_cutoff': 5,
                 'max_cutoff': 10, 'min_sigma': 0.1, 'max_sigma': 0.5}
    desc_dict_list = [descDict1, descDict2]

    pop_size = 20
    global conf_s, target_dataframe
    conf_s, target_dataframe = get_conf()
    # init_pop = initialise_gene_set(desc_dict_list, pop_size)
    # individual_pop = get_population(init_pop)
    # print([ind.score for ind in sorted(individual_pop)])
    sd = ['soap average cutoff=8 l_max=2 n_max=3 atom_sigma=0.31 n_Z=7 Z={8, 7, 6, 1, 16, 17, 9} n_species=7 species_Z={8, 7, 6, 1, 16, 17, 9} mu=0 mu_hat=0 nu=2 nu_hat=0', 'soap average cutoff=9 l_max=5 n_max=2 atom_sigma=0.36 n_Z=7 Z={8, 7, 6, 1, 16, 17, 9} n_species=7 species_Z={8, 7, 6, 1, 16, 17, 9} mu=0 mu_hat=0 nu=2 nu_hat=0']
    a, b = comp_soaps(sd, conf_s, target_dataframe)
    print(a.shape)

Main()
