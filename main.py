import numpy as np
import matplotlib.pyplot as plt
from iznetwork import IzNetwork 


class SmallWorldModular():
    """
    This class encapsulates the implementation details of a Modular Small World Network, initializes network parameters, 
    defines handler methods for initial connections setup and rewiring, as well as methods for connectivity matrices, 
    raster plots and mean firing rate plots.
    
    __init__(self) -- Initalizes parameter for the Modular Small World Network.
    
    initial_setup(self) -- Generates the modular network as per described experimental setup (Dynamical Complexity).
    
    apply_rewiring(self, W, D, p) -- Rewires existing connections with probability p.
    
    generate_connectivity_matrix(self, W) -- Creates connectivity matrix showcasing weights of connections (defined by W) between neurons.
    
    generate_raster_plot(self, firing_data) -- Generate raster plot to illustrate firing of the neurons. 
    
    generate_mean_firing_rate_plot(self) -- Generates the plot for mean firing rate of neurons per module in ms.    
    
    """
    def __init__(self):
        # Modular network parameters
        self.N = 1000                                                                                       # total number of neurons
        self.n_excitatory_neurons = 800                                                                     # excitatory neurons count
        self.n_inhibitory_neurons = self.N - self.n_excitatory_neurons                                      # inhibitory neurons count
        self.excitatory_neurons_per_module= 100
        self.inhibitory_neurons_in_module= 200
        self.n_excitatory_modules= int(self.n_excitatory_neurons / self.excitatory_neurons_per_module)      # no of excitatory modules
        self.n_inhibitory_module= int(self.n_inhibitory_neurons / self.inhibitory_neurons_in_module)        # no of inhibitiry modules
        self.total_modules= self.n_excitatory_modules + self.n_inhibitory_module                            # total modules

        self.Dmax = 20                                                                                      # maximum conduction delay in ms
        self.T = 1000                                                                                       # simulation time


    # Generate modular network as per experimental setup
    def initial_setup(self):
        """
        Initialise the modular network as per the required experimental setup
        delay. This involves initialization across 4 different possible combinations of connections.
        
        Excitatory-Excitatory : 1000 random connections within the module, Type : Small World, Modular, Weight = 1, Scaling factor = 17
                                Random delay from 1-20 ms(Dmax)
                                
        Excitatory-Inhibitory : Four excitatory neurons are connected to each inhibitory neuron, Type : Focal, Weight = Random(0,1), Scaling factor = 50
                                Fixed delay of 1 ms
        
        Inhibitory-Excitatory : All excitatory neurons in all modules, Type : Diffuse, Weight = Random(-1,0), Scaling factor = 2
                                Fixed delay of 1 ms
                                
        Excitatory-Excitatory : All inhibitory neurons, Type : Diffuse, Weight = Random(-1,0), Scaling factor = 1
                                Fixed delay of 1 ms

        Outputs:
        W  -- A numpy array for the  Weight, of size (N x N), N : Total number of neurons (1000). 

        D  -- A numpy array for the Conduction Delay, of size (N x N), N : Total number of neurons (1000).
        """
        
        W = np.zeros((self.N, self.N))                                                                        # weight matrix
        D = np.ones((self.N, self.N), dtype=int)                                                              # delay matrix initialized to ones (for zero delays)

        # Initialize excitatory-to-excitatory connections within modules
        for module_index in range(self.n_excitatory_modules):
            start_index = module_index * 100
            end_index = start_index + 100
            
            for _ in range(1000):                                                                   # 1000 random connections within the module
                src = np.random.randint(start_index, end_index)
                dest = np.random.randint(start_index, end_index)
                while src == dest:                                                                  # avoid self-connection
                    dest = np.random.randint(start_index, end_index)
                W[src, dest] = 1.0 * 17                                                             # unit weight scaled by 17
                D[src, dest] = np.random.randint(1, self.Dmax + 1)                                       # random delay from 1 to Dmax (always positive)

        # Intialize excitatory-to-inhibitory connections (focal, within module)
        for module_index in range(self.n_excitatory_modules):
            start_index = module_index * 100
            excitatory_start = start_index
            excitatory_end = start_index + 100
            inhibitory_start = self.n_excitatory_neurons + module_index * 25
            inhibitory_end = inhibitory_start + 25

            for inhibitory_neuron in range(inhibitory_start, inhibitory_end):
                # Connect each inhibitory neuron to 4 random excitatory neurons from same module
                exc_neurons_in_mod = np.random.choice(np.arange(excitatory_start, excitatory_end), 4, replace=False)
                for excitatory_neuron in exc_neurons_in_mod:
                    W[excitatory_neuron, inhibitory_neuron] = np.random.uniform(0, 1.0) * 50               # random weight between (0, 1) scaled by 50
                    D[excitatory_neuron, inhibitory_neuron] = 1                                     # delay is fixed at 1

        # Initialize inhibitory-to-excitatory connections (diffuse)
        for inhibitory_neuron in range(self.n_excitatory_neurons, self.N):
            for excitatory_neuron in range(self.n_excitatory_neurons):
                W[inhibitory_neuron, excitatory_neuron] = np.random.uniform(-1.0, 0.0) * 2               # random weight between (-1, 0) scaled by 2
                D[inhibitory_neuron, excitatory_neuron] = 1                                         # delay is fixed at 1

        # Initialize inhibitory-to-inhibitory connections (diffuse)
        for inhibitory_neuron1 in range(self.n_excitatory_neurons, self.N):
            for inhibitory_neuron2 in range(self.n_excitatory_neurons, self.N):
                if inhibitory_neuron1 != inhibitory_neuron2:
                    W[inhibitory_neuron1, inhibitory_neuron2] = np.random.uniform(-1.0, 0.0) * 1         # random weight (-1, 0) scaled by 1
                    D[inhibitory_neuron1, inhibitory_neuron2] = 1                                   # delay is fixed at 1
        
        return W, D


    # Rewire for small-world networks
    def apply_rewiring(self, W, D, p):
        """
        This method is used for re-wiring a given network, based on the rewiring probability 'p'.
         
        Inputs:
        W  -- Weight matrix, of size (N x N), N : Total number of neurons (1000). 

        D  -- Conduction Delay matrix, of size (N x N), N : Total number of neurons (1000).
        p  -- A real number in [0,1] as the Rewiring probability . 
        
        Outputs:
        W  -- Updated Weight matrix after rewiring, of size (N x N), N : Total number of neurons (1000). 

        D  -- Updated Conduction Delay matrix after rewiring, of size (N x N), N : Total number of neurons (1000).
        """
        for excitatory_module_index in range(self.n_excitatory_modules):                                                    # for each excitatory neuron module
            start_index = excitatory_module_index * self.excitatory_neurons_per_module
            end_index = start_index + self.excitatory_neurons_per_module
            
            # Random rewiring
            for i in range(start_index, end_index):
                for j in range(start_index, end_index):
                    if i!=j and W[i, j]> 0:
                        if np.random.rand() < p:
                            already_rewired=False
                            while not already_rewired:
                                target_module = np.random.choice([m for m in range(self.n_excitatory_modules) if m != excitatory_module_index])
                                target_start = target_module * self.excitatory_neurons_per_module
                                target_end = target_start + self.excitatory_neurons_per_module
                                dest = np.random.randint(target_start, target_end)

                                if W[i, dest] == 0:
                                    W[i, dest]= W[i, j]
                                    D[i, dest]= D[i, j]

                                    W[i, j]=0
                                    D[i, j]=1
                                    already_rewired=True    
        return W, D
    
    def generate_connectivity_matrix(self, W):
        """
        This method is used for plotting the connectivity matrix.
         
        Inputs:
        W  -- Weight matrix, of size (N x N), N : Total number of neurons (1000). 
        
        Outputs:
        Colorbar plot of the connectivity matrix
        """
        plt.figure(figsize=(8, 8))
        plt.imshow(W, cmap='binary', interpolation='none', vmin=0, vmax=1.5)
        plt.title(f"Connectivity Matrix (p = {p})")
        plt.xlabel("Neuron")
        plt.ylabel("Neuron")    
        plt.colorbar(label="Connection Weight")
        plt.ylim(self.n_excitatory_neurons, 0)
        plt.xlim(0, self.n_excitatory_neurons)
        plt.show()

    def generate_raster_plot(self, firing_data):
        """
        This method is used for raster plot of the neuron firing in a simulation run. 
         
        Inputs:
        firing_data  -- Firing data list comprising of (time-step, neuron_fired) tuples
                
        Outputs:
        Raster plot of the firing data for the given simulation.
        """
        if firing_data:
            times, neurons = zip(*firing_data)
            plt.figure(figsize=(12, 5))
            plt.scatter(times, neurons, s=16, color='blue', marker='o')
            plt.xlabel("Time (ms)")
            plt.ylabel("Neuron")
            plt.title(f"Raster Plot of Neuron Firing (p = {p})")
            plt.ylim(800, 0)
            plt.show()

    def generate_mean_firing_rate_plot(self):
        """
        This method is used for generating the mean firing rate plot in each module in a simulation run.     
               
        Outputs:
        Mean firing rate plot in each module.
        """
        module_sizes = [self.excitatory_neurons_per_module] * 8 + [self.inhibitory_neurons_in_module]                                                        # 8 - excitatory; 1 - inhibitory
        num_modules = len(module_sizes)
        firing_rates = np.zeros((num_modules, 50))                                              # 50 data points per module

        for i, (start, end) in enumerate(zip(range(0, self.T, 20), range(50, self.T + 50, 20))):
            # Find neurons firing in the current window
            window_firing = np.array([f for t, f in firing_data if start <= t < end])
        
            # Compute mean firing rate per module
            for module, size in enumerate(module_sizes):
                module_start = sum(module_sizes[:module])
                module_end = module_start + size
                module_firing = (window_firing >= module_start) & (window_firing < module_end)
                firing_rates[module, i] = np.sum(module_firing) / size  # Average firing per neuron

        # Plot mean firing rates for each module
        plt.figure(figsize=(12, 6))
        for module in range(num_modules):
            if module!=8:
                plt.plot(range(0, self.T, 20), firing_rates[module, :], label=f"Module {module + 1}")
        plt.xlabel("Time (ms)")
        plt.ylabel("Mean Firing Rate")
        plt.title(f"Mean Firing Rate in Each Module (p = {p})")
        # plt.legend()
        plt.show()



if __name__=="__main__":
    
    p_values=[0, 0.1, 0.2, 0.3, 0.4, 0.5]
    # p_values=[0.5]

    # For each rewiring probability
    for p in p_values:

        # Initialize Small World Modular Network
        smallWorldModular= SmallWorldModular()

        # Create IzNetwork object
        iznetwork= IzNetwork(smallWorldModular.N, smallWorldModular.Dmax)

        # Setup the initial connections
        W, D = smallWorldModular.initial_setup()

        # Define Izikevich neuron parameters
        r= np.random.random()
        a = np.concatenate([0.02 * np.ones(800), (0.02+0.08*r) * np.ones(200)])  # split parameters for excitatory & inhibitory
        b = np.concatenate([0.2 * np.ones(800), (0.25-0.05*r) * np.ones(200)])
        c = np.concatenate([(-65+15*(r**2)) * np.ones(800), -65 * np.ones(200)])
        d = np.concatenate([(8-6*(r**2))  * np.ones(800), 2 * np.ones(200)])

        # Set neuron parameters 
        iznetwork.setParameters(a, b, c, d)

        # Rewire excitatory-to-excitatory connections
        W, D= smallWorldModular.apply_rewiring(W, D, p)
        iznetwork.setWeights(W)                                                   # update network weights after rewiring 
        iznetwork.setDelays(D)                                                    # update network delays after rewiring

        # Plot the connectivity matrix
        smallWorldModular.generate_connectivity_matrix(W)

        # Run simulation 
        firing_data = [] 

        for t in range(smallWorldModular.T):
            # Inject background current using Poisson process
            background_current = (np.random.poisson(0.01, smallWorldModular.N) > 0) * 15
            iznetwork.setCurrent(background_current)

            # Update network and get neurons that fired
            fired = iznetwork.update()
            firing_data.extend([(t, neuron) for neuron in fired])

        # Raster plot of neuron firing
        smallWorldModular.generate_raster_plot(firing_data)

        # Mean firing rate in each module
        smallWorldModular.generate_mean_firing_rate_plot()
