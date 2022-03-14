# SOAP_GAS
A genetic algorithm to optimise the SOAP descriptor. 

## Motivation
The Smooth Overlap of Atomic Positions (SOAP) descriptor <REF> is a set of mathematical objects that can be used to represents and/or extract information from molecular structures. The SOAP descriptor has been used to build machine learning-based interatomic potentials for a number of materials. More relevant to the `SOAP_GAS` code, however, is the usage of the SOAP descriptor to predict the functional properties of molecular structures by means of machine learning algorithms. 
  
  At its core, the SOAP descriptor can be thought as a representation of the local atomic environments within a certain molecular structure. Said representation is obtained by using a local expansion of a Gaussian smeared atomic density with orthonormal functions based on spherical harmonics and radial basis functions. 
  
  In order to obtain a SOAP descriptor, one has to pick one or more atomic species as centre(s) of the local atomic environments and one or more species as the neighbors of the central atom that define said environment. It is not uncommon to use multiple SOAP descriptors, characterised by different choices of centres and neighbors, as they do contain information that might be hidden when simply choosing every atomic species in the molecule as both centre and neighbour. 
  
  A number of parameters are needed to define a SOAP descriptor, most prominently:
* **n_max**: The number of radial basis functions
* **l_max**: The maximum degree of the spherical harmonics
* **cutoff**: The spatial extent (in Å) of the local atomic environment
* **atom_sigma**: The standard deviation (in Å) of the Gaussian functions
  
  The choice of these parameters is not straightforward, and it is key to the accuracy and the predictive power of the SOAP descriptor. Physical intuition can provide a starting point, particularly in terms of the choice of `cutoff` (based on e.g. the extent of the moelcular structures in question), but finding (one of the) best combination(s) of these parameters often requires additional effort. A simple grid search in this 4-parameter space is a possibility, particularly if conducted in a randomised fashion, as the typical search space is simply too vast to be systematically explored. A typical search space would be defined by:
* 2 < **n_max** 10
* 2 < **l_max** 10
* 5 **cutoff** 20 [Å]
* 0.1 < **atom_sigma** < 1.5 [Å]
  The size of the grid obviously depends on the granularity of the variation with respect to each parameter, but typical sizes would ential be around 15,000 combinations.
  
  The computational effort required to assess the performance of a given SOAP descriptor varies according to the size of the dataset, the extent of the moelcular structures in question, and the number of centers and neighbors. In addition, different choices of the above mentioned four parameters will massively impact the resulting dimensionality of the SOAP vector, and thus the computational effort. Even when dealing with very small datasets (e.g. 100 molecules), a randomised grid search is needed to try and identify a sufficiently accurate combination of these SOAP parameters.
  
  In addition, optimising the SOAP parameters for different SOAPS via a randomised grid search *at the same time* would involve an intractacble number of potential combinations.
  
## The Genetic Algorithm
  The `SOAP_GAS` code seeks to optimise the SOAP parameters by means of a genetic algorithm, which structure is depicted in the figure below. We start by constructing a so-called Initial Population containing a certain number of Individuals. Each Individual is SOAP descriptor characterised by a randomly selected set of SOAP parameters. For each Invidual within the Initial Population we compute a Score, that is a metric that quantify the accuracy of the Individual in predicting the functional property of interest. The choice of the specific machine learning algorithm used to compute the Score can be easily modified by leveraging widely used Python packages such as [scikit-learn](https://scikit-learn.org/stable/). When dealing with regression problems, the Score involves the mean squared error (MSE) and the Pearson correlation coefficient (PCC) for both training and test test, whilst for classification problems the Score is based on the Matthews correlation coefficient (MCC) - see Ref.<REF> for the details. A certain number (`bestSample`, see the input reference )
  
  Then you choose the best bestSample plus some luckyFew. This selection you breed. Choose random pairs and then each pair has one offspring where you choose random parameters from each parent, 50% chance from A or B parent. Then, mutation, that is, we are go to change each parameter by means of mutationChance (individually). Then the breeding is repeated numberChildren times. Which means: ((bestSample+luckyFew)/2)*numberChildren = population size. 
  
  dealing with RAM issues. Original implementation: empty list, 1st iiteration populates the list (12 individuals). 2nd generation + more invididuals. The `individual` class contains:
  - 4 parameters
  - SOAP vector
  - target values for the whole database
  - score
  - each of the K splits train/test for the inputs 
  For each generation, this class is written to disk to free RAM

  
  
## Content
The `SOAP_GAS` code contains a genetic algorithm that enables the optimisation of the four SOAP parameters introduced in the previous section - given a certain dataset of molecular structures and their corresponding functional properties.
  
## Features
* Regression as well as classification capabilities
* Full control of the search space
* Full control of the genetic algorithm parameters
* Heterogeneous datasets containing different moelcules with different numbers of atomic species can be considered via the average keyword (see <REF>)
* 3D molecular models of crystals, liquids or amorphous systems can also be considered
* Compressed SOAP descriptors can be used to reduce the dimensionality of the descriptors (and thus the computational effort), see <REF>
* Simultaneous optimisation of the SOAP parameters for different SOAP descriptors at the same time
  
## Installation
The `SOAP_GAS` code is written in Python (3.x) and leverages several Python packages. The full list of pre-requisites can be found in <>. Most notably 
  ### LiNuX
  * gcc 4.9 (or later/higher)
  * GA: git clone https://github.com/TrentBarnard/TrentPhD.git (SOAP_GAS)
  * QUIP: git clone --recursive https://github.com/libAtoms/QUIP.git
  * Get rid of the original src dir in the QUIP: cd QUIP/src ; rm -rf GAP; 
  * Get the compression stuff: git clone --recursive https://github.com/JPDarby/GAP.git ; cd GAP; git checkout compression ; 
  * module load GCC/10.2.0 OpenMPI/4.0.5 SciPy-bundle/2020.11 # matplotlib 
  * cd ../../ $ export QUIP_ARCH=linux_x86_64_gfortran (darwin for mac) ;
  * -lopenblas
  * make config
  * make
  * make quippy 
  * QUIPPY_INSTALL_OPTS=--user make install-quippy
  
 Some python packages
  - Python/3.8.6
  - SciPy-bundle/2020.11 (Bottleneck 1.3.2
deap 1.3.1
mpi4py 3.0.3
mpmath 1.1.0
numexpr 2.7.1
numpy 1.19.4
pandas 1.1.4
scipy 1.5.4)
  - We need: ase-3.22.1
  - pip install -U scikit-learn (scikit-learn-1.0.2)
  - python genAlg.py input !!!
  ### MacOS
  ### HPC facilities
  
## Input reference
  
  The atoms present in your dataset are (excluding H): {'H', 'C', 'N', 'S', 'CL', 'O', 'F', 'P'}
The atom counts are: {'H': 2552, 'C': 2161, 'O': 419, 'N': 232, 'S': 39, 'CL': 38, 'F': 31, 'P': 1}
The number of molecules that contain each atom are: {'H': 131, 'C': 131, 'O': 127, 'N': 92, 'S': 30, 'CL': 27, 'F': 13, 'P': 1}
The total number of molecules in the dataframe is: 131
What is the maximum number of atoms for the possible centre/neighbour sets? (Enter an int between 1 and 8): 4
Which atoms (if any) would you like to drop? Please seperate them by spaces
C
The combinations that you could use as centre/neighbour atoms are: [('H',), ('H', 'N'), ('H', 'S'), ('H', 'CL'), ('H', 'O'), ('H', 'F'), ('H', 'P'), ('N', 'O'), ('H', 'N', 'S'), ('H', 'N', 'CL'), ('H', 'N', 'O'), ('H', 'N', 'F'), ('H', 'N', 'P'), ('H', 'S', 'CL'), ('H', 'S', 'O'), ('H', 'S', 'F'), ('H', 'S', 'P'), ('H', 'CL', 'O'), ('H', 'CL', 'F'), ('H', 'CL', 'P'), ('H', 'O', 'F'), ('H', 'O', 'P'), ('H', 'F', 'P'), ('N', 'S', 'O'), ('N', 'CL', 'O'), ('N', 'O', 'F'), ('N', 'O', 'P'), ('H', 'N', 'S', 'CL'), ('H', 'N', 'S', 'O'), ('H', 'N', 'S', 'F'), ('H', 'N', 'S', 'P'), ('H', 'N', 'CL', 'O'), ('H', 'N', 'CL', 'F'), ('H', 'N', 'CL', 'P'), ('H', 'N', 'O', 'F'), ('H', 'N', 'O', 'P'), ('H', 'N', 'F', 'P'), ('H', 'S', 'CL', 'O'), ('H', 'S', 'CL', 'F'), ('H', 'S', 'CL', 'P'), ('H', 'S', 'O', 'F'), ('H', 'S', 'O', 'P'), ('H', 'S', 'F', 'P'), ('H', 'CL', 'O', 'F'), ('H', 'CL', 'O', 'P'), ('H', 'CL', 'F', 'P'), ('H', 'O', 'F', 'P'), ('N', 'S', 'CL', 'O'), ('N', 'S', 'O', 'F'), ('N', 'S', 'O', 'P'), ('N', 'CL', 'O', 'F'), ('N', 'CL', 'O', 'P'), ('N', 'O', 'F', 'P')]
  
  
The specifics of the SOAP descriptor(s) to be optimised are contained in one or more Python dictionaries in the following form:
SOAP_1_dictionary = {'lower' : int ,'upper' : int, 'centres' : '{int(, int, ..., int)}', 'neighbours' : '{(, int, ..., int)}', 'mu' : int, 'mu_hat': int, 'nu': int, 'nu_hat': int, 'average': Boolean, 'mutationChance': float, 'min_cutoff': float, 'max_cutoff' : float, 'min_sigma': float, 'max_sigma': float}
  where:
  * `'lower'` : int. Lower bound for both n_max and l_max 
  * `'upper'` : int. Upper limit for both n_max and l_max
  * `'centres'` : '{int(, int, ..., int)}'. atomi numbers for each centre.neigh
  * `'neighbours'` : '{(, int, ..., int)}'
  - reference to James's paper for explanation about the compresison keys
  * `'mu'` : int
  * `'mu_hat'` : int
  * `'nu'`: int
  * `'nu_hat'`: int
  * `'average'`: Boolean
  - reference to Albert's paper
  * `'mutationChance'`: float. probability that one of the SOAP parameteres will mutate from one generation to the next. The same probability is applied independently to each parameter.
  * `'min_cutoff'`: float
  * `'max_cutoff'` : float
  * `'min_sigma'`: float
  * `'max_sigma'`: float

For instance: `descDict3 = {'lower' : 2,'upper' : 6,'centres' : '{8, 7, 6, 1, 16, 17, 9}','neighbours' : '{8, 7, 6, 1, 16, 17, 9}','mu' : 0,'mu_hat': 0, 'nu':2, 'nu_hat':0, 'average':True, 'mutationChance': 0.15, 'min_cutoff':5, 'max_cutoff' : 10, 'min_sigma':0.1, 'max_sigma':0.5})`
* `descList` ([descDict3])
* `numberOfGenerations` (20)
* `popSize` (12)
* `bestSample` (6) bestSample+luckyFew cannot be an odd number because of the breeding
* `luckyFew` (2)
* `numberChildren` (3)
NOPE * `multi` (True) NOPE
  
Initial population (different SOAP parameters initialised randomly). Get a score for each. Then you choose the best bestSample plus some luckyFew. This selection you breed. Choose random pairs and then each pair has one offspring where you choose random parameters from each parent, 50% chance from A or B parent. Then, mutation, that is, we are go to change each parameter by means of mutationChance (individually). Then the breeding is repeated numberChildren times. Which means: ((bestSample+luckyFew)/2)*numberChildren = population size. 
  
## Output files
* Backed up outputs
* out_<input_file_name>.txt
* best_<input_file_name>.pkl (list of individual classes but no vector and only for the best ones / generation) . learning curves and stuff
* history_<input_file_name>.pkl (list of list of individual classes / generation)-> this can be massive!
  
## Workflow
* atomicStats.py: returns information about the frequency by which a given atomic species is contained within the dataset
* genAlg.py : optimises the SOAP descriptor parameters for one or multiple SOAPs
* gasVisual.ipynb : visualization of the key results within a Jupyter Notebook
  
## Examples
### C-C SOAP optimisation
### all-all SOAP optimisation, with compression
### Multiple SOAP simultaneous optimisation
