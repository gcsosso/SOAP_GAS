import pandas as pd


from dataclasses import dataclass
from quippy import descriptors
import ase.io
import subprocess
from pathlib import Path
import os
import pickle as pkl
from random import sample
import numpy as np

@dataclass
class GeneParameters:
    lower: int
    upper: int
    centres: str
    neighbours: str
    mu: int
    mu_hat: int
    nu: int
    nu_hat: int
    mutation_chance: float
    min_cutoff: int
    max_cutoff: int
    min_sigma: float
    max_sigma: float
class GeneSet(GeneParameters):
    def __init__(self,
                cutoff: int,
                l_max: int,
                n_max: int,
                sigma: int):
                lower: int,
                upper: int,
                centres: str,
                neighbours: str,
                mu: int,
                mu_hat: int,
                nu: int,
                nu_hat: int,
                mutation_chance: float,
                min_cutoff: int,
                max_cutoff: int,
                min_sigma: float,
                max_sigma: float):
        super().__init__(gene_parameters)
        self.cutoff = cutoff
        self.l_max = l_max
        self.n_max = n_max
        self.sigma = sigma
    def __eq__(self, other):
        return self.__dict__==other.__dict__


    def mutate_gene(self):
        new_mutation = comp_random_genes(**desc_dict)
        possible_mutations = ['cutoff, l_max', 'n_max', 'sigma']
        for poss in possible_mutations:
            rand = np.random.rand()
            if rand > self.mutation_chance:
                self.__dict__[poss] = new_mutation.__dict__[poss]
                print(f'{poss} has mutated')

    def breed_genes(self, other):
        pass



class Individual:
    def __init__(self, gene_set_list):
        self.gene_set_list = gene_set_list
        self.soap_string_list = [get_parameter_string(gene_set) for gene_set in gene_set_list]
        soaps, targets = comp_soaps(self.soap_string_list, conf_s, target_dataframe)
        self.score = comp_score(soaps, targets)

    def __le__(self, other):
        return self.score <= other.score

    def __ge__(self, other):
        return self.score >= other.score

    def __lt__(self, other):
        return self.score < other.score

    def __gt__(self, other):
        return self.score > other.score

    def __eq__(self, other):
        return self.soap_string_list==other.soap_string_list

    def __hash__(self):
        return hash(tuple(self.soap_string_list))

    def breed(self, other):
        pass


def comp_score(soaps, targets):
    return int(soaps)


def comp_random_genes(set_of_gene_parameters: GeneParameters) -> GeneSet:
    """
    This takes the parameters given in the input file and returns an instance of
    an Genes class with random parameters.
    """
    gene_parameters = kwargs
    gene_parameters['cutoff'] = np.random.randint(min_cutoff, max_cutoff)
    gene_parameters['l_max'] = np.random.randint(lower, upper)
    gene_parameters['n_max'] = np.random.randint(lower, upper)
    gene_parameters['sigma'] = round(np.random.uniform(min_sigma, max_sigma), 2)
    return GeneSet(**gene_parameters)


def initialise_gene_set(pop_size: int) -> list[GeneSet]:
    """
    Creates the initial population of size popSize by generating a GeneSet
    for each set of parameters in the input file
    """
    return [[comp_random_genes(**desc_dict) for desc_dict in desc_dict_list] for _ in range(pop_size)]


def get_population(list_of_Genes: list[list[GeneSet]]) -> list[Individual]:
    """
    This function creates a population of individuals based on the genes given as input
    """
    return [Individual(gene_set_list) for gene_set_list in list_of_Genes]


def comp_soaps(parameter_string_list: list[str], conf_s: list, data: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """
    Function to calculate the SOAP feature vector using the parameters of each individual
    """
    # soap_array = []
    # targets = np.array(data["Target"])
    # for mol in conf_s:
    #     soap = []
    #     for parameter_string in parameter_string_list:
    #         soap += list(descriptors.Descriptor(parameter_string).calc(mol)['data'][0])
    #     soap_array.append(soap)
    # return np.array(soap_array), np.array(targets)
    tempVal = 0
    for parameter_string in parameter_string_list:
        tempVal += int(parameter_string[20]) + int(parameter_string[28]) + int(parameter_string[36])
    return np.array([tempVal]), np.array([9])


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


def next_generation(previous_generation: list[Individual],
                    best_sample: int,
                    lucky_few: int,
                    number_of_children: int) -> list[Individual]:
    # Delete duplicate individuals and replace them with new random ones
    unique_individuals = set(previous_generation)
    print(f"{len(previous_generation) - len(unique_individuals)} duplicate individuals found")
    while len(previous_generation)!=len(unique_individuals):
        unique_individuals.add(Individual([comp_random_genes(**desc_dict) for desc_dict in desc_dict_list]))
    print("Non unique individuals replaced with new individuals")
    sorted_individuals = sorted(list(unique_individuals))
    parents = select_from_population(sorted_individuals, best_sample, lucky_few)

    return sorted_individuals


def breed(parents: list[Individual]) -> list[Individual]:
    children = []
    shuffle(parents)
    it = iter(parents)
    # children.append(it.)
    pass
def select_from_population(sorted_individuals: list[Individual], best_sample: int, lucky_few: int) -> list[Individual]:
    best_individuals = sorted_individuals[:best_sample]
    lucky_individuals = sample(sorted_individuals[best_sample:], lucky_few)
    # write to file "THE BEST INDIVIDUALS ARE
    return best_individuals + lucky_individuals


def Main():
    descDict1 = {'lower': 2, 'upper': 3, 'centres': '{8, 7, 6, 1, 16, 17, 9}', 'neighbours': '{8, 7, 6, 1, 16, 17, 9}',
                 'mu': 0, 'mu_hat': 0, 'nu': 2, 'nu_hat': 0, 'average': True, 'mutationChance': 0.15, 'min_cutoff': 5,
                 'max_cutoff': 10, 'min_sigma': 0.1, 'max_sigma': 0.1}
    descDict2 = {'lower': 2, 'upper': 3, 'centres': '{8, 7, 6, 1, 16, 17, 9}', 'neighbours': '{8, 7, 6, 1, 16, 17, 9}',
                 'mu': 0,
                 'mu_hat': 0, 'nu': 2, 'nu_hat': 0, 'average': True, 'mutationChance': 0.15, 'min_cutoff': 5,
                 'max_cutoff': 10, 'min_sigma': 0.1, 'max_sigma': 0.1}

    pop_size = 20
    global conf_s, target_dataframe, desc_dict_list
    desc_dict_list = [descDict1, descDict2]
    conf_s, target_dataframe = get_conf()
    init_pop = initialise_gene_set(pop_size)
    print(init_pop[0][1])
    individual_pop = get_population(init_pop)
    print([ind.score for ind in sorted(individual_pop)])
    print([ind.__hash__() for ind in sorted(individual_pop)])
    next_gen = next_generation(individual_pop, 1, 2, 3)
    print(len(next_gen))
    # sd = ['soap average cutoff=8 l_max=2 n_max=3 atom_sigma=0.31 n_Z=7 Z={8, 7, 6, 1, 16, 17, 9} n_species=7 species_Z={8, 7, 6, 1, 16, 17, 9} mu=0 mu_hat=0 nu=2 nu_hat=0', 'soap average cutoff=9 l_max=5 n_max=2 atom_sigma=0.36 n_Z=7 Z={8, 7, 6, 1, 16, 17, 9} n_species=7 species_Z={8, 7, 6, 1, 16, 17, 9} mu=0 mu_hat=0 nu=2 nu_hat=0']
    # a, b = comp_soaps(sd, conf_s, target_dataframe)
    # print(a.shape)
    # dict_genes1 = {'centres': '{8, 7, 6, 1, 16, 17, 9}', 'neighbours': '{8, 7, 6, 1, 16, 17, 9}', 'mu': 0, 'mu_hat': 0,
    #                'nu': 2, 'nu_hat': 0, 'average': True, 'mutationChance': 0.15, 'cutoff': 7, 'l_max': 4, 'n_max': 5,
    #                'sigma': 0.48}
    # dict_genes2 = {'centres': '{6, 7, 6, 1, 16, 17, 9}', 'neighbours': '{8, 7, 6, 1, 16, 17, 9}', 'mu': 0, 'mu_hat': 0,
    #                'nu': 2, 'nu_hat': 0, 'average': True, 'mutationChance': 0.15, 'cutoff': 7, 'l_max': 4, 'n_max': 5,
    #                'sigma': 0.48}
    # test_gene1 = Genes(**dict_genes1)
    # test_gene2 = Genes(**dict_genes2)
    # gene_set_list = [test_gene1, test_gene2]
    # individual1 = Individual(gene_set_list)
    # individual2 = Individual(gene_set_list)
    # print(individual1 == individual2)
    # print(vars(individual1))


# Main()
descDict2 = {'lower': 2, 'upper': 3, 'centres': '{8, 7, 6, 1, 16, 17, 9}', 'neighbours': '{8, 7, 6, 1, 16, 17, 9}',
                 'mu': 0,
                 'mu_hat': 0, 'nu': 2, 'nu_hat': 0, 'mutation_chance': 0.15, 'min_cutoff': 5,
                 'max_cutoff': 10, 'min_sigma': 0.1, 'max_sigma': 0.1}

a = GeneParameters(**descDict2)
print(vars(a))
b = GeneSet(1,2,3,4,a)
print(vars(b))
# a = comp_random_genes(**descDict2)
# a.mutate_gene()