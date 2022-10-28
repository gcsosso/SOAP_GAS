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
    """
    Class to save the parameters given as inputs
    ...
    Attributes
    ----------
    lower : int
        The lower bound for l_max and n_max
    upper : int
        The upper bound for l_max and n_max
    centres : str
        A string of the format '{a, b, c}' where a, b and c are all
        chemical symbols that represent the species to be used as centres
    neighbours : str
        A string of the format '{a, b, c}' where a, b and c are all
        chemical symbols that represent the species to be used as neighbours
    mu : int
        Compression parameter
    mu_hat : int
        Compression parameter
    nu : int
        Compression parameter
    nu_hat : int
        Compression parameter
    mutation_chance : float
        Probability of a gene mutating after breeding
    min_cutoff : int
        Lower bound for cutoff
    max_cutoff : int
        Upper bound for cutoff
    min_sigma : float
        Lower bound for sigma
    max_sigma : float
        Upper bound for sigma

    Methods
    -------
    make_gene_set()
        Generates a random set of genes in the form of a GeneSet class
    """
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
        """ Generates a random set of genes in the form of a GeneSet class

        Returns
        -------
        GeneSet
            A GeneSet class that has been generated using random values
            within the bounds stipulated in this GeneParameter class
        """
        cutoff = np.random.randint(self.min_cutoff, self.max_cutoff)
        l_max = np.random.randint(self.lower, self.upper)
        n_max = np.random.randint(self.lower, self.upper)
        sigma = round(np.random.uniform(self.min_sigma, self.max_sigma), 2)
        return GeneSet(self, cutoff, l_max, n_max, sigma)


class GeneSet:
    """
    A class used to represent a specific set of genes

    ...

    Attributes
    ----------
    gene_parameters : GeneParameter
        A GeneParameter class that contains information about the
        parameters used to generate this GeneSet
    cutoff : int
        The cutoff function for the basis function (Å)
    l_max : int
        The maximum degree of the spherical harmonics
    n_max : int
        The number of radial basis functions
    sigma : float
        The Gaussian spearing width of atom density σ (Å)

    Methods
    -------
    mutate_gene()
        Mutates the GeneSet instance
    get_soap_string()
        Returns a string that can be used as an input to generate SOAPs
    """

    def __init__(self, gene_parameters, cutoff, l_max, n_max, sigma):
        self.gene_parameters = gene_parameters
        self.cutoff = cutoff
        self.l_max = l_max
        self.n_max = n_max
        self.sigma = sigma

    def __str__(self):
        return f"[{self.cutoff}, {self.l_max}, {self.n_max}, {self.sigma}]"

    def __repr__(self):
        return f"GeneSet({self.cutoff}, {self.l_max}, " \
               f"{self.n_max}, {self.sigma})"

    def mutate_gene(self):
        """ Mutates the GeneSet instance

        Each GeneSet has a mutation chance that is inherited from the
        GeneParameter class used to create the GeneSet. This method does
        not return anything and instead permanently modifies the GeneSet
        instance that runs it.
        """
        new_mutation = self.gene_parameters.make_gene_set()
        weights = [1 - self.gene_parameters.mutation_chance,
                   self.gene_parameters.mutation_chance]
        self.cutoff = choices([self.cutoff, new_mutation.cutoff],
                              weights=weights, k=1)[0]
        self.l_max = choices([self.l_max, new_mutation.l_max],
                             weights=weights, k=1)[0]
        self.n_max = choices([self.n_max, new_mutation.n_max],
                             weights=weights, k=1)[0]
        self.sigma = choices([self.sigma, new_mutation.sigma],
                             weights=weights, k=1)[0]

    def get_soap_string(self):
        """Returns a string that can be used as an input to generate SOAPs

        This function calculates the number of centre and neighbour atoms
        and then uses the parameters of the class to generate a string that
        can be used to generate a SOAP descriptor.

        Returns
        -------
        str
            A string that contains all the parameters required to create a
            SOAP descriptor array
        """
        num_centre_atoms = sum(c.isdigit() for c in
                               self.gene_parameters.centres)

        num_neighbour_atoms = sum(n.isdigit() for n in
                                  self.gene_parameters.neighbours)
        return "soap average cutoff={cutoff} l_max={l_max} n_max={n_max} " \
               "atom_sigma={sigma} n_Z={0} Z={centres} " \
               "n_species={1} species_Z={neighbours} mu={mu} mu_hat={" \
               "mu_hat} nu={nu} nu_hat={nu_hat}".format(
            num_centre_atoms, num_neighbour_atoms,
            **{**vars(self.gene_parameters), **vars(self)})


class Individual:
    """
    A class that represents an Individual. An individual is made of a
    collection of GeneSet classes.

    ...

    Attributes
    ----------
    gene_set_list : list[GeneSet]
        A list of GeneSet instances
    score : float
        The overall score of the individual
    soap_string_list : list[str]
        A list of strings where the strings are used to create a SOAP array

    Methods
    -------
    get_score()
        Updates the individuals score
    """

    def __init__(self, gene_set_list):
        self.gene_set_list = gene_set_list
        self.score = 0
        self.soap_string_list = [gene_set.get_soap_string() for gene_set in
                                 gene_set_list]

    def __str__(self):
        return f"Individual(" \
               f"{[str(gene_set) for gene_set in self.gene_set_list]})"

    def __repr__(self):
        return f"Individual" \
               f"({[repr(gene_set) for gene_set in self.gene_set_list]})"

    def __le__(self, other):
        return self.score <= other.score

    def __ge__(self, other):
        return self.score >= other.score

    def __lt__(self, other):
        return self.score < other.score

    def __gt__(self, other):
        return self.score > other.score

    def __eq__(self, other):
        return self.gene_set_list == other.gene_set_list

    def __hash__(self):
        return hash(tuple(self.soap_string_list))

    def get_score(self):
        """Updates the Individuals score

        This can be computationally expensive, so this method is only
        called when it is necessary to access the score of the individual.
        This method does not return anything and instead updates the score
        attribute.
        """
        # Actual function commented out while testing
        self.score = self.gene_set_list[0].cutoff + self.gene_set_list[
            1].cutoff
        # soaps, target_dataframergets = comp_soaps(self.soap_string_list,
        # conf_s, target_dataframe)
        # self.score = comp_score(soaps, targets)


class Population:
    """
    A class that represents the entire population of Individuals

    ...

    Attributes
    ----------
    best_sample : int
        The number of best scoring Individuals that are used as breeders
        for the next generation
    lucky_few : int
        The number of lucky Individuals that are used as breeders for the
        next generation. These can not be any of the Individuals that are
        selected as the best Individuals
    population_size : int
        The total size of the population
    number_of_children : int
        The number of children to be produced from breeding
    list_of_gene_parameters : list[GeneSet]
        A list of the GeneSet instances used to create this Individual
    maximise_scores : bool
        Determines whether the SOAP_GAS algorithm will minimise or maximise
        scores (default False)
    population : set[Individuals]
        The entire population of individuals that make up the current
        generation of the SOAP_GAS algorithm
    Methods
    -------
    print_population()
        Prints the population to the console in a way that is easy to read
    initialise_population()
        Initialises the population attribute
    next_generation()
        Updates the class with a new random population
    get_population_scores()
        Gets the scores for all the Individuals in the population
    """

    def __init__(self, best_sample, lucky_few, population_size,
                 number_of_children, list_of_gene_parameters,
                 maximise_scores=False):
        self.best_sample = best_sample
        self.lucky_few = lucky_few
        self.number_of_children = number_of_children
        self.population_size = population_size
        self.list_of_gene_parameters = list_of_gene_parameters
        self.maximise_scores = maximise_scores
        self.population = set()

    def __repr__(self):
        return f"Population({self.best_sample}, {self.lucky_few}, " \
               f"{self.population_size}, {self.number_of_children}, " \
               f"{self.list_of_gene_parameters}, {self.maximise_scores})"

    def __eq__(self, other):
        return (self.best_sample==other.best_sample and
                self.lucky_few==other.lucky_few and
                self.number_of_children==other.number_of_children and
                self.population_size==other.population_size and
                self.list_of_gene_parameters==
                other.list_of_gene_parameters and
                self.maximise_scores==other.maximise_scores)

    def print_population(self):
        """Prints the population to the console in a way that is easy to read

        This method does not return anything and instead prints text to
        the console.
        """
        for ind in self.population:
            print(f"{ind} has a score of: {ind.score}")

    def initialise_population(self):
        """Initialises the population attribute

        This is done by creating unique random Individual instances and
        adding them to the population attribute until the population
        reaches the specified size. This method does not return anything
        and instead modifies the population attribute.
        """
        self.population = set()
        while len(self.population) < self.population_size:
            gene_set_list = [gene_parameters.make_gene_set() for
                             gene_parameters in self.list_of_gene_parameters]
            self.population.add(Individual(gene_set_list))
        self.get_population_scores()
        print(f"Initial population of size {self.population_size} generated")

    def next_generation(self):
        """Updates the class with a new random population

        This is done by breeding a set of Individuals comprised of a
        specified number of the best Individuals and lucky Individuals.

        The protocol for generating the next generation of Individuals is
        as follows:
        1) The specified number of the best Individuals are selected and put
        into a breeding pool.
        2) The specified number of lucky Individuals are selected and added
        to the breeding pool. NOTE: These can not be selected if they are
        already in the breeding pool
        3) Initially, the Individuals in the breeding pool are paired up
        with the higher scoring Individuals being partners with each other
        4) Each set of partners produces one child Individual and adds it
        to the population for the next generation. If the child
        Individual already exists in the new population, this set of
        partners will continue to produce children and try to add it to the
        new population until a unique child is added.
        5) Once all sets of partners have produced one child, the breeding
        pool is shuffled. This means that for the creation of future
        children, it is not necessarily the best performing Individuals
        that breed with each other, Individuals in the breeding pool are
        paired up randomly.
        6) Loop back to step 4) and repeat the process until the population
        is the same size as it was before breeding occurred.
        7) Get the scores for all the Individuals in the new population.
        This is done now because if the scores were calculated upon
        creation of a new individual, computational power would be wasted
        on duplicate Individuals.
        """
        sorted_population = sorted(self.population,
                                   reverse=self.maximise_scores)
        best_individuals = sorted_population[:self.best_sample]
        lucky_individuals = sample(sorted_population[self.best_sample:],
                                   self.lucky_few)
        next_generation_parents = best_individuals + lucky_individuals
        self.population = set()
        for _ in range(self.number_of_children):
            it = iter(next_generation_parents)
            while True:
                try:
                    parent_one = next(it)
                    parent_two = next(it)
                    child = breed_individuals(parent_one, parent_two)
                    while child in self.population:
                        child = breed_individuals(parent_one, parent_two)
                    self.population.add(child)
                except StopIteration:
                    break
        shuffle(next_generation_parents)
        self.get_population_scores()

    def get_population_scores(self):
        """Gets the scores for all the Individuals in the population


        Updates all the Individuals in the population attribute with their
        scores. This can be computationally expensive, so it is only
        performed when necessary. Ths method does not return anything and
        instead updates the population attribute.
        """
        for individual in self.population:
            individual.get_score()

    def sort_population(self):
        """Sorts the population

        This method does not return anything and instead modifies the
        population attribute.
        """
        self.population = sorted(self.population,
                                 reverse=self.maximise_scores)


class BestHistory:
    def __init__(self, early_stop=None, early_number=None,
                 min_generations=None):
        self.history = []
        self.converged = False
        self.early_stop = early_stop
        self.early_number = early_number
        self.min_generations = min_generations

    def append(self, population):
        if not isinstance(population, Population):
            raise TypeError("You can only append a Population to the "
                            "BestHistory")
        if self.history:
            if not self.history[-1] == population.population[0]:
                raise TypeError("Trying to append population of different "
                                "type")
        population.sort_population()
        self.history.append(population.population[0])
        self._check_if_converged()

    def _check_if_converged(self):
        if not (self.early_stop and self.early_number and
                self.min_generations):
            return
        if self.min_generations <= self.early_number:
            raise ValueError("The minimum number of 'converged' generations"
                             " must be less than the total number "
                             "of generations.")
        if len(self.history) < self.min_generations:
            return
        try:
            maximise_scores = self.history[0].maximise_scores
            sorted_history = sorted(self.history, reverse=maximise_scores)
        except:
            return
        best_score = sorted_history[0]
        if all(best_score.score - ind.score <= self.early_stop for ind in
               sorted_history[:self.early_number]):
            print("SOAP_GAS has converged")
            self.converged = True




def breed_genes(gene_set_one, gene_set_two):
    """Function to breed two GeneSets

    Breeds two GeneSet instances with each other to create a new GeneSet
    instance that is a combination of both the parents.

    Parameters
    ----------
    gene_set_one : GeneSet
        The first GeneSet instance to be used for breeding
    gene_set_two : GeneSet
        The second GeneSet instance to be used for breeding

    Returns
    -------
    GeneSet
        A GeneSet instance that is a combination of both parents

    Raises
    ------
    TypeError
        If two GeneSets that were created using different GeneParameter
        values, they will not be able to breed with each other
    """
    if gene_set_one.gene_parameters!=gene_set_two.gene_parameters:
        raise TypeError("Trying to breed genes with different "
                        "gene parameters")

    cutoff = choice([gene_set_one.cutoff, gene_set_two.cutoff])
    l_max = choice([gene_set_one.l_max, gene_set_two.l_max])
    n_max = choice([gene_set_one.n_max, gene_set_two.n_max])
    sigma = choice([gene_set_one.sigma, gene_set_two.sigma])
    new_gene_set = GeneSet(gene_set_one.gene_parameters, cutoff,
                           l_max, n_max, sigma)
    new_gene_set.mutate_gene()
    return new_gene_set


def breed_individuals(individual_one, individual_two):
    """Function to breed two Individuals

    Breeds two Individual instances with each other to create a new Individual
    instance that is a combination of both the parents.

    Parameters
    ----------
    individual_one : Individual
        The first GeneSet instance to be used for breeding
    individual_two : Individual
        The second GeneSet instance to be used for breeding

    Returns
    -------
    Individual
        An Individual instance that is a combination of both parents
    """
    new_gene_set_list = []
    for genes in zip(individual_one.gene_set_list,
                     individual_two.gene_set_list):
        new_gene_set_list.append(breed_genes(genes[0], genes[1]))
    return Individual(new_gene_set_list)


def comp_score(soaps, targets):
    """
    Function to compute the score given soaps and targets. Still needs to
    be done
    @param soaps:
    @param targets:
    @return:
    """
    return int(soaps)


def comp_soaps(parameter_string_list, conf_s, data):
    """
    Function to compute the Soaps, maybe add to the Individual class?
    """
    pass
    # Actual function commented out while testing that everything works.
    # soap_array = []
    # targets = np.array(data["Target"])
    # for mol in conf_s:
    #     soap = []
    #     for parameter_string in parameter_string_list:
    #         soap += list(descriptors.Descriptor(parameter_string).calc(
    #         mol)['data'][0])
    #     soap_array.append(soap)
    # return np.array(soap_array), np.array(targets)


def get_conf():
    """Function to get the inputs required to get SOAP arrays

    This function checks if the conf_s file exists, returns the ase
    objects and a dataframe of targets if conf_s exists.
    If it does not exist, creates the conf_s file and returns the same thing.

    Returns
    -------

    """
    path = str(os.path.dirname(os.path.abspath(__file__))) + "/EXAMPLES/"
    if os.path.isfile(path + "conf_s.pkl"):
        print("conf_s file exists")
        return [*pkl.load(open(path + "conf_s.pkl", "rb"))]
    else:
        try:
            csv_path = Path(path + "database.csv").resolve(strict=True)
            xyz_folder_path = Path(path + "xyz/").resolve(strict=True)
        except FileExistsError:
            print("Please make sure the xyz folder and database.csv file "
                  "exist. Read the docs for more information.")
        print("Generating conf")
        names_and_targets = pd.read_csv(csv_path)
        conf_s = []
        for index, row in names_and_targets.iterrows():
            xyzname = str(xyz_folder_path) + '/' + row['Name'] + '.xyz'
            subprocess.call(
                "sed 's/CL/Cl/g' " + xyzname + " | sed 's/LP/X/g' > tmp.xyz",
                shell=True)
            conf = ase.io.read("tmp.xyz")
            conf_s.append(conf)
        subprocess.call("rm tmp.xyz", shell=True)
        print(f"saving conf to {path}conf_s.pkl")
        pkl.dump([conf_s, names_and_targets], open(path + 'conf_s.pkl', 'wb'))
        return [conf_s, names_and_targets]
