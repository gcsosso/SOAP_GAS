import numpy as np

descDict1 = {'lower': 2, 'upper': 6, 'centres': '{8, 7, 6, 1, 16, 17, 9}', 'neighbours': '{8, 7, 6, 1, 16, 17, 9}',
             'mu': 0, 'mu_hat': 0, 'nu': 2, 'nu_hat': 0, 'average': True, 'mutationChance': 0.15, 'min_cutoff': 5,
             'max_cutoff': 10, 'min_sigma': 0.1, 'max_sigma': 0.5}
descDict2 = {'lower': 8, 'upper': 12, 'centres': '{8, 7, 6, 17, 9}', 'neighbours': '{8, 7, 17, 9}', 'mu': 0,
             'mu_hat': 0, 'nu': 2, 'nu_hat': 0, 'average': True, 'mutationChance': 0.15, 'min_cutoff': 5,
             'max_cutoff': 10, 'min_sigma': 0.1, 'max_sigma': 0.5}
desc_list = [descDict1, descDict2]


################################################################################
class GeneSet:
    def __init__(self, cutoff: int, l_max: int, n_max: int, sigma: int) -> None:
        self.cutoff = cutoff
        self.l_max = l_max
        self.n_max = n_max
        self.sigma = sigma


def comp_random_genes(lower: int,
                      upper: int,
                      min_cutoff: int,
                      max_cutoff: int,
                      min_sigma: float,
                      max_sigma: float) -> GeneSet:
    """
    This takes the parameters given in the input file and returns an instance of
    an GeneSet class with random parameters.
    """
    cutoff = np.random.randint(min_cutoff, max_cutoff)
    l_max = np.random.randint(lower, upper)
    n_max = np.random.randint(lower, upper)
    sigma = round(np.random.uniform(min_sigma, max_sigma), 2)
    return GeneSet(cutoff, l_max, n_max, sigma)


def initialise_individuals(desc_list: list[dict], pop_size: int) -> list:
    """
    Creates the initial population of size popSize by generating a GeneSet
    for each set of parameters in the input file
    """
    return [[comp_random_genes(**descDict) for descDict in desc_list] for _ in range(pop_size)]


a = initialise_individuals(desc_list, 20)
print(a)