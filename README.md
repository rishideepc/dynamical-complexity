# About
This coursework report entails simulation of a modular small world network of `Izhikevich` neuron model, in Python programming language. We run simulations on the network and study the effects of various rewiring probabilities on the results of these simulations. We do so with the help of Connectivity matrices, Raster plots and Mean Firing Rate plots.

# How to run code
Python3 is needed to run this experiment. First, clone the repository and navigate to the root folder:
```bash
$ git clone https://github.com/rishideepc/dynamical-complexity.git
$ cd dynamical-complexity
```
Now that we are in the root folder, we have to execute the following:
```bash
$ python3 main.py
```
Running the above command will display the connectivity matrices, raster plots and the plots for mean-firing-rates with the rewiring probability $P$ , $\forall P \in ${0, 0.1, 0.2, 0.3, 0.4, 0.5} in this order.

# Code Structure
This section provides a brief overview of the implementation of the modular small world network of `Izhikevich` neurons in `main.py`
  #### _SmallWorldModular_ class
This class encapsulates the implementation details of a Modular Small World Network, initializes network parameters, defines handler methods for initial connections setup and rewiring, as well as methods for connectivity matrices, raster plots and mean firing rate plots.
  #### _main_ method
This method is respoonsible for creating the objects of `SmallWorldNetwork` and `IzNetwork` classes, and simulate the experiment to display the result.

# Imports and Libraries
We have utilized `numpy` for mathematical operations, and `matplotlib` to generate the plots. The class `iznetwork.py` defines the implementation of Izhikevich neuron model class and it has been imported to implement the same in the code.
  
