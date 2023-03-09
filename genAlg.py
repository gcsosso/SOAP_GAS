import pandas as pd
from sklearn.model_selection import RepeatedKFold
from dataclasses import dataclass
from quippy import descriptors
import ase
import sys
import subprocess
from pathlib import Path
import os
import pickle as pkl
from random import sample, shuffle, choice, choices
import numpy as np
from collections import defaultdict
from sklearn.preprocessing import MinMaxScaler
import tensorflow
from tensorflow.keras import layers, optimizers, Model, backend
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
from scipy.stats import pearsonr
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from ase.geometry.analysis import Analysis

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# TODO: Make number of message steps global, not a GeneParameter input
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
    nu_S : int
        Compression parameter
    nu_R : int
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
    nu_R: int
    nu_S: int
    mutation_chance: float
    min_cutoff: int
    max_cutoff: int
    min_sigma: float
    max_sigma: float
    message_steps: int

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
        num_centre_atoms = sum(c.strip().isdigit() for c in
                               self.gene_parameters.centres[
                               1:-1].split(','))

        num_neighbour_atoms = sum(n.strip().isdigit() for n in
                                  self.gene_parameters.neighbours[
                                  1:-1].split(','))
        if self.gene_parameters.message_steps==0:
            return "soap average cutoff={cutoff} l_max={l_max} " \
                   "n_max={n_max} atom_sigma={sigma} n_Z={0} Z={centres} " \
                   "n_species={1} species_Z={neighbours} nu_R={nu_R} " \
                   "nu_S={nu_S}".format(
                    num_centre_atoms, num_neighbour_atoms,
                    **{**vars(self.gene_parameters), **vars(self)})
        elif self.gene_parameters.message_steps > 0:
            return "soap cutoff={cutoff} l_max={l_max} n_max={n_max} " \
                   "atom_sigma={sigma} n_Z={0} Z={centres} " \
                   "n_species={1} species_Z={neighbours} nu_R={nu_R} " \
                   "nu_S={nu_S}".format(
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
    check_same_gene_parameters(other : Individual)
        Checks if other is an Individual created from the same gene parameters
    """

    def __init__(self, gene_set_list):
        self.gene_set_list = gene_set_list
        self.score = 0
        self.soap_string_list = [gene_set.get_soap_string() for gene_set in
                                 gene_set_list]
        self.soaps = None
        self.targets = None
        self.regression = None
        self.results_dictionary = defaultdict(list)

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
        conf_s, data = get_conf()
        column_names = data.columns
        try:
            if column_names[1].upper().strip() == 'TARGET':
                self.regression = True
            elif column_names[1].upper().strip() == 'CLASS':
                self.regression = False
        except:
            raise TypeError("Your database.csv file isc not of the correct "
                             "format. Please read the docs.")
        self.comp_soaps(conf_s, data)
        # Add to parameter file
        splits = 5
        repeats = 1
        self.cv_split(splits, repeats, regression=self.regression)

    def check_same_gene_parameters(self, other):
        """Checks if other is an Individual created from the same gene parameters

        Parameters
        ----------
        other : Individual
            The Individual to compare gene parameters

        Returns
        -------
        bool
            Returns True if both Individuals are created from the same list of GeneParameter classes,
            otherwise returns False
        """
        for pair in zip(self.gene_set_list, other.gene_set_list):
            if pair[0].gene_parameters != pair[1].gene_parameters:
                return False
        return True

    def comp_soaps(self, conf_s, data):
        """
        Function to compute the Soaps, maybe add to the Individual class?
        """
        soap_array = []
        targets = data.iloc[:, 1].values

        if self.gene_set_list[0].gene_parameters.message_steps > 0:
            try:
                for mol in conf_s:
                    soap = []
                    for parameter_string in self.soap_string_list:
                        a_soap = descriptors.Descriptor(parameter_string).calc(
                                 mol)['data']
                        mp_soap = np.mean(multiple_message_passes(mol, self.gene_set_list[0].gene_parameters.message_steps) @ a_soap, axis=0)
                        soap += list(mp_soap)
                    soap_array.append(soap)
            except:
                soap = []
                for parameter_string in self.soap_string_list:
                    try:
                        a_soap = descriptors.Descriptor(parameter_string).calc(
                                mol)['data']
                        mp_soap = np.mean(multiple_message_passes(mol, self.gene_set_list[0].gene_parameters.message_steps) @ a_soap, axis=0)
                        soap += list(mp_soap)
                    except:
                        a_soap = descriptors.Descriptor(parameter_string).calc(
                                mol)['data']
                        mp_matrix = multiple_message_passes(mol, self.gene_set_list[0].gene_parameters.message_steps)
                        print("Mismatch in MP and SOAP matrix shapes for: "
                              , mol)
                        print("SOAP shape of: {}".format(a_soap.shape))
                        print("MP matrix shape of: {}".format(
                            mp_matrix.shape))
                        print("The SOAP string is: {}".format(
                            parameter_string))
        else:
            for mol in conf_s:
                # print(f"Getting soap for {mol}")
                soap = []
                for parameter_string in self.soap_string_list:
                    soap += list(descriptors.Descriptor(
                        parameter_string).calc(mol)['data'][0])
                soap_array.append(soap)
        self.soaps = np.array(soap_array)
        self.targets = np.array(targets)

    def add_to_results_dictionary(self, results):
        for k, v in results:
            self.results_dictionary[k].append(v)

    def cv_split(self, splits, repeats, random_state=999,
                 regression=None):
        """
        Returns split indices for train and test sets
        """
        cv = RepeatedKFold(n_splits=splits, n_repeats=repeats,
                           random_state=random_state)
        if regression is not None:
            if regression:
                for train_index, test_index in cv.split(self.soaps):
                     self.get_scores_regression(train_index,
                                                test_index)
            elif not regression:
                # encoder = LabelEncoder()
                # self.targets = encoder.fit_transform(self.targets)
                for train_index, test_index in cv.split(self.soaps):
                    self.get_scores_classification(train_index,
                                                    test_index)
            self.score = np.average(self.results_dictionary["scores"])

    def get_scores_regression(self, train_index, test_index):
        estimator = build_model(self.soaps, regression=True)
        X_train, X_test, X_scaler = scale_data(self.soaps[train_index],
                                               self.soaps[test_index])
        y_train, y_test, y_scaler = scale_data(
            self.targets[train_index].reshape(-1, 1),
            self.targets[test_index].reshape(-1, 1))

        res = scorer_NN_regression(estimator, X_train, X_test, y_train,
                                   y_test, y_scaler)
        self.add_to_results_dictionary(res)

    def get_scores_classification(self, train_index, test_index):
        estimator = build_model(self.soaps, regression=False)
        Y = self.targets
        encoder = LabelEncoder()
        encoder.fit(Y)
        encoded_Y = encoder.transform(Y)
        y = to_categorical(encoded_Y)

        X_train, X_test, X_scaler = scale_data(self.soaps[train_index],
                                               self.soaps[test_index])
        y_train, y_test = y[train_index], y[test_index]
        res = scorer_NN_class(estimator, X_train, X_test, y_train, y_test)
        self.add_to_results_dictionary(res)

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
    sort_population()
        Sorts the population attribute based on best score
    """

    def __init__(self, best_sample, lucky_few, population_size,
                 number_of_children, list_of_gene_parameters,
                 maximise_scores=False, **kwargs):
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
        return (self.best_sample == other.best_sample and
                self.lucky_few == other.lucky_few and
                self.number_of_children == other.number_of_children and
                self.population_size == other.population_size and
                self.list_of_gene_parameters ==
                other.list_of_gene_parameters and
                self.maximise_scores == other.maximise_scores)

    def print_population(self):
        """Prints the population to the console in a way that is easy to read

        This method does not return anything and instead prints text to
        the console
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
        write_to_outfile(f"Initial population of size {self.population_size} generated")

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
        counter = 1
        for individual in self.population:
            write_to_outfile(f"Getting score for individual {counter} of "
                  f"{self.population_size}")
            counter += 1
            individual.get_score()
            write_to_outfile(f"Score: {individual.score}")

    def sort_population(self):
        """Sorts the population

        This method does not return anything and instead modifies the
        population attribute.
        """
        self.population = sorted(self.population,
                                 reverse=self.maximise_scores)


class BestHistory:
    """
    A class that stores the best Individuals after each generation

    ...

    Attributes
    ----------
    history : list[Individual]
        A list of the Individual with the best score after each generation
    converged : bool
        True if the SOAP_GAS algorithm has converged
    early_stop : float
        The threshold used to test convergence
    early_number : int
        The number of generations that must converge before stopping
    min_generations : int
        The minimum number of generations before stopping

    Methods
    -------
    append()
        Used to append the best Individual of a Population to the history
    _check_if_converged()
        Tests to see if the algorithm has converged (hidden method)
    """

    def __init__(self, early_stop=None, early_number=None,
                 min_generations=None):
        self.history = []
        self.converged = False
        self.early_stop = early_stop
        self.early_number = early_number
        self.min_generations = min_generations

    def append(self, population):
        """Method to append the best Individual of a Population to the history attribute

        This method does not return anything, but it adds an Individual to the history attribute.

        Raises
        ------
        TypeError
            If a class that is not a Population is appended to the history attribute
        TypeError
            If a Population that is formed using different gene_parameters is appended to the history attribute
        ValueError
            If min_generations is <= early_number

        @param population:
        @return:
        """
        if not isinstance(population, Population):
            raise TypeError("You can only append a Population to the "
                            "BestHistory")
        if self.history:
            if not self.history[-1].check_same_gene_parameters(list(population.population)[0]):
                raise TypeError("Trying to append population of different "
                                "type")
        population.sort_population()
        best_ind = population.population[0]
        self.history.append(best_ind)
        write_to_outfile(f"Best Individual {str(best_ind)} with a score of {best_ind.score} added to history")
        self._check_if_converged(population.maximise_scores)

    def _check_if_converged(self, maximise_scores):
        if not (self.early_stop and self.early_number and
                self.min_generations):
            return
        if self.min_generations <= self.early_number:
            raise ValueError("The minimum number of 'converged' generations"
                             " must be less than the total number "
                             "of generations.")
        if len(self.history) < self.min_generations:
            return
        sorted_history = sorted(self.history, reverse=maximise_scores)
        best_score = sorted_history[0]
        if all(best_score.score - ind.score <= self.early_stop for ind in
               sorted_history[:self.early_number]):
            print("Converged")
            write_to_outfile("SOAP_GAS has converged")
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
    if gene_set_one.gene_parameters != gene_set_two.gene_parameters:
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

def get_conf():
    """Function to get the inputs required to get SOAP arrays

    This function checks if the conf_s file exists, returns the ase
    objects and a dataframe of targets if conf_s exists.
    If it does not exist, creates the conf_s file and returns the same thing.

    Returns
    -------

    """
    path = str(os.path.dirname(os.path.abspath(__file__))) + \
           "/EXAMPLES/"
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
            xyz_name = str(xyz_folder_path) + '/' + row['Name'] + '.xyz'
            subprocess.call(
                "sed 's/CL/Cl/g' " + xyz_name + " | sed 's/LP/X/g' > tmp.xyz",
                shell=True)
            conf = ase.io.read("tmp.xyz")
            conf_s.append(conf)
        subprocess.call("rm tmp.xyz", shell=True)
        print(f"saving conf to {path}conf_s.pkl")
        pkl.dump([conf_s, names_and_targets], open(path + 'conf_s.pkl', 'wb'))
        return [conf_s, names_and_targets]


# get adjacency matrix, input has to be ase object
# selfconnect = False, returns adjacency matrix
# selfconnect = True, returns adjacency matrix with self connections
def adjmat(aseobj,selfconnect=False):
    from ase import neighborlist
    na=len(aseobj)
    nl=neighborlist.build_neighbor_list(aseobj)
    conmat=nl.get_connectivity_matrix(sparse=False)
    adj=np.triu(conmat,1)+np.tril(conmat,-1)
    if selfconnect:
        adj=adj+np.transpose(adj)+np.eye(na).astype(int)
    if not selfconnect:
        adj=adj+np.transpose(adj)
    return adj


# Get matrix for a single message pass, input has to be an ase object
def message_passing_matrix(mol):
    # adjacency with self connections
    adjacency_matrix = adjmat(mol,selfconnect=True)
    # matrix that averages the atomic features
    diagonal_matrix = np.zeros_like(adjacency_matrix)
    np.fill_diagonal(diagonal_matrix, adjacency_matrix.sum(axis=1))
    averaged_adjacency_matrix = np.linalg.inv(diagonal_matrix)@adjacency_matrix
    return averaged_adjacency_matrix


# Get matrix for N message passes
def multiple_message_passes(mol, N):
    print(f"Doing multiple message passes {N} times")
    return np.linalg.matrix_power(message_passing_matrix(mol), N)


# Return descriptor for molecule using average of atomic features
def mol_descriptor(mol, N):
    atom_features = []
    for atom in mol.GetAtoms():
        atom_features.append(atom_featurizer.encode(atom))
    atom_features = np.array(atom_features)
    return np.mean(multiple_message_passes(mol, N) @ atom_features, axis=0)

def scorer_NN_regression(estimator, X_train, X_test, y_train, y_test, y_scaler):
    """ Scoring function for use with NN regressor. Added by Matt. """

    callback = EarlyStopping(monitor='val_loss', patience=50)
    estimator.fit(X_train, y_train, callbacks=[callback], validation_data =
    (X_test, y_test), epochs=200, verbose=False)
    y_test_pred, y_train_pred = estimator.predict(X_test, verbose=False), estimator.predict(X_train, verbose=False)
    y_test_pred, y_train_pred = y_scaler.inverse_transform(y_test_pred), y_scaler.inverse_transform(y_train_pred)
    y_test = y_scaler.inverse_transform(y_test.reshape(-1, 1))
    y_train = y_scaler.inverse_transform(y_train.reshape(-1, 1))
    y_test = np.ravel(y_test)
    y_train = np.ravel(y_train)
    y_train_pred = y_train_pred.ravel()
    y_test_pred = y_test_pred.ravel()
    testCorr = pearsonr(y_test, y_test_pred)[0]
    trainCorr = pearsonr(y_train, y_train_pred)[0]
    testMSE = mean_squared_error(y_test, y_test_pred)
    trainMSE = mean_squared_error(y_train, y_train_pred)
    score = (2 * (trainMSE * (1-trainCorr)) + (testMSE * (1-testCorr)))
    res = [('scores', score), ('y_train_actual', y_train), ('y_test_actual', y_test), ('y_train_predictions', y_train_pred), ('y_test_predictions', y_test_pred)]
    return res

def scorer_NN_class(estimator, X_train, X_test, y_train, y_test):
    """ Scoring function for use with NN classifier. Added by Trent. """
    callback = EarlyStopping(monitor='val_loss', patience=50)
    estimator.fit(X_train, y_train, callbacks=[callback], validation_data=(
        X_test, y_test), epochs=200, verbose=True)
    y_test_pred = estimator.predict(X_test)
    y_train_pred = estimator.predict(X_train)
    _, test_accuracy = estimator.evaluate(X_test, y_test)
    _, train_accuracy = estimator.evaluate(X_train, y_train)
    y_test_pred = np.argmax(y_test_pred, axis=1)
    y_train_pred = np.argmax(y_train_pred, axis=1)
    y_test_actual = np.argmax(y_test, axis=1)
    y_train_actual = np.argmax(y_train, axis=1)
    score = -1 * (test_accuracy + (0.5 * train_accuracy))
    res = [('scores', score), ('y_train_actual', y_train_actual), \
          ('y_test_actual', y_test_actual), ('y_train_predictions', y_train_pred),
               ('y_test_predictions', y_test_pred)]
    return res

def scale_data(train, test):
    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train)
    test_scaled = scaler.transform(test)

    return train_scaled, test_scaled, scaler


def build_model(X, regression=None):
    backend.clear_session()
    tensorflow.random.set_seed(12345)
    input_layer = Input(X.shape[1])
    hidden_layer = input_layer
    for layer in [100,100,100]:
        hidden_layer = Dense(layer, activation='relu')(hidden_layer)
    if regression is not None:
        if regression:
            output_layer = Dense(units=1, activation='linear')(hidden_layer)
            model = Model(input_layer, output_layer)
            model.compile(loss='mean_squared_error', optimizer=optimizers.Adam(learning_rate=0.01), metrics=['mean_squared_error'])
        elif not regression:
            output_layer = Dense(units=3, activation='softmax')(hidden_layer)
            model = Model(input_layer, output_layer)
            model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model
    else:
        raise ValueError("Regression or classification can not be determined")


if __name__ == '__main__':
    def check_outfile_exists(file_path):
        if os.path.exists(file_path):
            print("Making Backup for {}".format(file_path))
            numb = 1
            while True:
                new_path = "{}-Backup{}.pkl".format(file_path[:-4], numb)
                if os.path.exists(new_path):
                    numb += 1
                else:
                    break
            os.rename(file_path, new_path)
            print("Backed up {} to {}".format(file_path, new_path))
            return

    def write_to_outfile(words):
        outputFile = str(os.path.dirname(
            os.path.abspath(__file__))) + "/out_{}.txt".format(sys.argv[1])
        with open(outputFile, 'a+') as f:
            f.write(words)
            f.write('\n')
        return


    input_file_name = sys.argv[1]

    check_outfile_exists(str(os.path.dirname(os.path.abspath(__file__))) +
               "/history_{}.pkl".format(input_file_name))
    check_outfile_exists(str(os.path.dirname(os.path.abspath(__file__))) +
               "/out_{}.txt".format(input_file_name))

    input_parameters = __import__(input_file_name)

    write_to_outfile("Starting genetic algorithm with the following "
                     "parameters:")
    write_to_outfile(str(input_parameters.population_parameters))
    write_to_outfile(str(input_parameters.history_parameters))
    write_to_outfile("The GA will run for a maximum of "
                       f"{input_parameters.num_gens} generations")
    conf_s, data = get_conf()
    column_names = data.columns
    try:
        if column_names[1].upper().strip() == 'TARGET':
            write_to_outfile("Type of machine learning: Regression")
        elif column_names[1].upper().strip() == 'CLASS':
            write_to_outfile("Type of machine learning: Classification")
    except:
        write_to_outfile("Error with database.csv")
        raise ValueError("Your database.csv file is not of the correct "
                         "format. Please read the docs.")

    gene_parameters = [GeneParameters(**params) for params in
                       input_parameters.descList]
    pop = Population(**input_parameters.population_parameters,
                     list_of_gene_parameters=gene_parameters)
    hist = BestHistory(**input_parameters.history_parameters)

    pop.initialise_population()
    for gen in range(input_parameters.num_gens):
        if hist.converged:
            break
        print(f"Generation {gen}")
        write_to_outfile(f"Generation {gen}")
        pop.next_generation()
        hist.append(pop)
        write_to_outfile("-------")
    pkl.dump(hist, open(str(os.path.dirname(os.path.abspath(__file__))) +
             "/history_{}.pkl".format(input_file_name), 'wb'))
    write_to_outfile("History has been saved")
