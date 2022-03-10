# SOAP_GAS
A genetic algorithm to optimise the SOAP descriptor. 

# Motivation
The Smooth Overlap of Atomic Positions (SOAP) descriptor <REF> is a set of mathematical objects that can be used to represents and/or extract information from molecular structures. The SOAP descriptor has been used to build machine learning-based interatomic potentials for a number of materials. More relevant to the `SOAP_GAS` code, however, is the usage of the SOAP descriptor to predict the functional properties of molecular structures by means of machine learning algorithms. 
  
  At its core, the SOAP descriptor can be thought as a representation of the local atomic environments within a certain moelcular structure. Said representation is obtained by using a local expansion of a Gaussian smeared atomic density with orthonormal functions based on spherical harmonics and radial basis functions. 
  
  In order to obtain a SOAP descriptor, one has to pick one or more atomic species as centre(s) of the local atomic environments and one or more species as the neighbors of the central atom that define said environment. It is not uncommon to use multiple SOAP descriptors, characterised by different choices of centres and neighbors, as they do contain information that might be hidden when simply choosing every atomic species in the molecule as both centre and neighbour. 
  
  A number of parameters are needed to define a SOAP descriptor, most prominently:
* **n_max**: The number of radial basis functions
* **l_max**: The maximum degree of the spherical harmonics
* **cutoff**: The spatial extent (in Å) of the local atomic environment
* **atom_sigma**: The standard deviation (in Å) of the Gaussian functions
  
  The choice of these parameters is not straightforward, and it is key to the accuracy and the predictive power of the SOAP descriptor. Physical intuition can provide a starting point, particularly in terms of the choice of *cutoff* (based on e.g. the extent of the moelcular structures in question), but finding (one of the) best combination(s) of these parameters often requires additional effort. A simple grid search in this 4-parameter space is a possibility, particularly if conducted in a randomised fashion, as the typical search space is simply too vast to be systematically explored. A typical search space would be defined by:
* 2 < **n_max** 10
* 2 < **l_max** 10
* 5 **cutoff** 20 [Å]
* 0.1 < **atom_sigma** < 1.5 [Å]
  The size of the grid obviously depends on the granularity of the variation with respect to each parameter, but typical sizes would ential be around 15,000 combinations.
  
  The computational effort required to assess the performance of a given SOAP descriptor varies according to the size of the dataset, the extent of the moelcular structures in question, and the number of centers and neighbors. In addition, different choices of the above mentioned four parameters will massively impact the resulting dimensionality of the SOAP vector, and thus the computational effort. Even when dealing with very small datasets (e.g. 100 molecules), a randomised grid search is needed to try and identify a sufficienlty accurate combination of these SOAP parameters.
  
  In addition, optimising the SOAP parameters for different SOAPS via a randomised grid search at the same time would involve an intractacble number of potential combinations.
  
  The `SOAP_GAS` code seeks to optimise the SOAP parameters by means of a genetic algorithm, the details of which are discussed in <REF>

# Content
The `SOAP_GAS` code contains a genetic algorithm that enables the optimisation of the four SOAP parameters introduced in the previous section - given a certain dataset of molecular structures and their corresponding functional properties.
  
# Features
* Regression as well as classification capabilities
* Full control of the search space
* Full control of the genetic algorithm parameters
* Heterogeneous datasets containing different moelcules with different numbers of atomic species can be considered via the average keyword (see <REF>)
* 3D molecular models of crystals, liquids or amorphous systems can also be considered
* Compressed SOAP descriptors can be used to reduce the dimensionality of the descriptors (and thus the computational effort), see <REF>
* Simultaneous optimisation of the SOAP parameters for different SOAP descriptors at the same time
  
# Installation
  
# Workflow
  
# Examples
