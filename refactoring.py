import pandas as pd

from dataclasses import dataclass
from quippy import descriptors
import ase.io
import subprocess
from pathlib import Path
import os
import pickle as pkl
from random import sample, shuffle, choice, choices
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

    def make_gene_set(self):
        cutoff = np.random.randint(self.min_cutoff, self.max_cutoff)
        l_max = np.random.randint(self.lower, self.upper)
        n_max = np.random.randint(self.lower, self.upper)
        sigma = round(np.random.uniform(self.min_sigma, self.max_sigma), 2)
        return GeneSet(self, cutoff, l_max, n_max, sigma)


class GeneSet():
    def __init__(self, gene_parameters, cutoff, l_max, n_max, sigma):
        self.gene_parameters = gene_parameters
        self.cutoff = cutoff
        self.l_max = l_max
        self.n_max = n_max
        self.sigma = sigma

    def __str__(self):
        return f"[{self.cutoff}, {self.l_max}, {self.n_max}, {self.sigma}]"

    def __repr__(self):
        return f"GeneSet({self.cutoff}, {self.l_max}, {self.n_max}, {self.sigma})"

    def mutate_gene(self):
        new_mutation = self.gene_parameters.make_gene_set()
        mutation_chance = self.gene_parameters.mutation_chance
        self.cutoff = choices([self.cutoff, new_mutation.cutoff], weights=[1 - mutation_chance, mutation_chance], k=1)[0]
        self.l_max = choices([self.l_max, new_mutation.l_max], weights=[1 - mutation_chance, mutation_chance], k=1)[0]
        self.n_max = choices([self.n_max, new_mutation.n_max], weights=[1 - mutation_chance, mutation_chance], k=1)[0]
        self.sigma = choices([self.sigma, new_mutation.sigma], weights=[1 - mutation_chance, mutation_chance], k=1)[0]
    def breed(self, other):
        # print(f"before breed {self}")
        self.cutoff = choice([self.cutoff, other.cutoff])
        self.l_max = choice([self.l_max, other.l_max])
        self.n_max = choice([self.n_max, other.n_max])
        self.sigma = choice([self.sigma, other.sigma])
        self.mutate_gene()
        # print(f"after breed {self}")
        return self


class Individual:
    def __init__(self, gene_set_list):
        self.gene_set_list = gene_set_list
        self.soap_string_list = [get_parameter_string(gene_set) for gene_set in gene_set_list]
        self.score = 0

    def __str__(self):
        return f"Individual({[str(gene_set) for gene_set in self.gene_set_list]})"

    def __repr__(self):
        return f"({[repr(gene_set) for gene_set in self.gene_set_list]})"

    def __le__(self, other):
        return self.score <= other.score

    def __ge__(self, other):
        return self.score >= other.score

    def __lt__(self, other):
        return self.score < other.score

    def __gt__(self, other):
        return self.score > other.score

    def __eq__(self, other):
        return self.soap_string_list == other.soap_string_list

    def __hash__(self):
        return hash(tuple(self.soap_string_list))

    def breed(self, other):
        for genes in zip(self.gene_set_list, other.gene_set_list):
            print(f"breeding {genes}")
            genes[0].breed(genes[1])
            print(f"created {genes}")
        return Individual(self.gene_set_list)

    def get_score(self):
        self.score = self.gene_set_list[0].cutoff + self.gene_set_list[1].cutoff
        # soaps, target_dataframergets = comp_soaps(self.soap_string_list, conf_s, target_dataframe)
        # self.score = comp_score(soaps, targets)


class Population:
    def __init__(self, best_sample, lucky_few, population_size, number_of_children, list_of_gene_parameters):
        self.best_sample = best_sample
        self.lucky_few = lucky_few
        self.number_of_children = number_of_children
        self.population_size = population_size
        self.list_of_gene_parameters = list_of_gene_parameters
        self.population = set()

    def initialise_population(self):
        while len(self.population) < self.population_size:
            gene_set_list = [gene_parameters.make_gene_set() for gene_parameters in self.list_of_gene_parameters]
            self.population.add(Individual(gene_set_list))
        print(f"Initial population of size {self.population_size} generated")

    def next_generation(self):
        sorted_population = sorted(self.population)
        best_individuals = sorted_population[:self.best_sample]
        lucky_individuals = sample(sorted_population[self.best_sample:], self.lucky_few)
        next_generation_parents = best_individuals + lucky_individuals
        self.population = set()
        for _ in range(self.number_of_children):
            shuffle(next_generation_parents)
            it = iter(next_generation_parents)  # Creates an iterator from randomly shuffled parents
            while True:
                try:
                    parent_one = next(it)
                    parent_two = next(it)
                    child = parent_one.breed(parent_two)
                    # print(f"Breeding {parent_one} with {parent_two}")
                    # print(f"Created child {child}")
                    # Tries to create a child and add it to the population. If the child already exists in the
                    # population then it creates another child until the new child does not exist in the population.
                    while child in self.population:
                        child = parent_one.breed(parent_two)
                        print("CHILD ALREADY EXISTS")
                        break
                    self.population.add(child)
                except StopIteration:
                    break
        self.get_population_scores()
        for ind in self.population:
            print(f"{ind} has a score of {ind.score}")
    def get_population_scores(self):
        for individual in self.population:
            individual.get_score()


def comp_score(soaps, targets):
    return int(soaps)


# def comp_random_genes(set_of_gene_parameters: GeneParameters) -> GeneSet:
#     """
#     This takes the parameters given in the input file and returns an instance of
#     an Genes class with random parameters.
#     """
#     gene_parameters = kwargs
#     gene_parameters['cutoff'] = np.random.randint(min_cutoff, max_cutoff)
#     gene_parameters['l_max'] = np.random.randint(lower, upper)
#     gene_parameters['n_max'] = np.random.randint(lower, upper)
#     gene_parameters['sigma'] = round(np.random.uniform(min_sigma, max_sigma), 2)
#     return GeneSet(**gene_parameters)


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
    num_centre_atoms = gene_set.gene_parameters.centres.count(',') + 1
    num_neighbour_atoms = gene_set.gene_parameters.neighbours.count(',') + 1
    return "soap average cutoff={cutoff} l_max={l_max} n_max={n_max} atom_sigma={sigma} n_Z={0} Z={centres} " \
           "n_species={1} species_Z={neighbours} mu={mu} mu_hat={mu_hat} nu={nu} nu_hat={nu_hat}".format(
        num_centre_atoms, num_neighbour_atoms, **{**vars(gene_set.gene_parameters), **vars(gene_set)})


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
    while len(previous_generation) != len(unique_individuals):
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
    descDict1 = {'lower': 2, 'upper': 10, 'centres': '{8, 7, 6, 1, 16, 17, 9}', 'neighbours': '{8, 7, 6, 1, 16, 17, 9}',
                 'mu': 0, 'mu_hat': 0, 'nu': 2, 'nu_hat': 0, 'average': True, 'mutationChance': 0.15, 'min_cutoff': 5,
                 'max_cutoff': 10, 'min_sigma': 0.1, 'max_sigma': 0.9}
    descDict2 = {'lower': 2, 'upper': 10, 'centres': '{8, 7, 6, 1, 16, 17, 9}', 'neighbours': '{8, 7, 6, 1, 16, 17, 9}',
                 'mu': 0,
                 'mu_hat': 0, 'nu': 2, 'nu_hat': 0, 'average': True, 'mutationChance': 0.15, 'min_cutoff': 5,
                 'max_cutoff': 10, 'min_sigma': 0.1, 'max_sigma': 0.9}

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
descDict1 = {'lower': 1, 'upper': 50, 'centres': '{8, 7, 6, 1, 16, 17, 9}', 'neighbours': '{8, 7, 6, 1, 16, 17, 9}',
             'mu': 0, 'mu_hat': 0, 'nu': 2, 'nu_hat': 0, 'mutation_chance': 0.0, 'min_cutoff': 1,
             'max_cutoff': 50, 'min_sigma': 0.1, 'max_sigma': 0.9}
descDict2 = {'lower': 51, 'upper': 100, 'centres': '{8, 7, 6, 1, 16, 17, 9}', 'neighbours': '{8, 7, 6, 1, 16, 17, 9}',
             'mu': 0,
             'mu_hat': 0, 'nu': 2, 'nu_hat': 0, 'mutation_chance': 0.0, 'min_cutoff': 51,
             'max_cutoff': 100, 'min_sigma': 1.1, 'max_sigma': 1.9}

params1 = GeneParameters(**descDict1)
params2 = GeneParameters(**descDict2)
population = Population(2, 1, 5, 5, [params1, params2])
print(population.population)
population.initialise_population()
print(population.population)
population.next_generation()
# a = comp_random_genes(**descDict2)
# a.mutate_gene()
