import numpy as np
import matplotlib.pyplot as plt
from numpy import pi
from mpl_toolkits.axes_grid1 import make_axes_locatable
import random
from matplotlib import cm
import matplotlib

"""
 applying Metropolis algorithm for XY-Model
 input: T/temperature
        S/spins configuration(in 1d list)
"""


class XYSystem():
    def __init__(self, temperature, thetas, supp_end, boundary):
        self.N = thetas.shape[0]
        self.L = int(thetas.shape[0] ** (1/2))
        self.support_end = supp_end
        L, N = self.L, self.N
        self.nbr = {i: ((i // L) * L + (i + 1) % L, (i + L) % N,
                    (i // L) * L + (i - 1) % L, (i - L) % N) \
                                            for i in list(range(N))}
        self.thetas = thetas
        self.alpha_config = self._transf(self.thetas, - pi, pi, 0, self.support_end)
        self.temperature = temperature
        self.energy = np.sum(self.get_energy()) / self.N
        self.M = []
        self.Cv = []
        self.boundary = boundary

    
    def sweep(self):
        beta = 1.0 / self.temperature
        alpha_idx = list(range(self.N))
        random.shuffle(alpha_idx)
        acceptance = np.random.uniform(size=(self.N, self.alpha_config.shape[1]), low=0.0, high=1.0)
        prop = np.random.uniform(size=(self.N, self.alpha_config.shape[1]), low=- self.boundary, high = self.boundary)
        for idx in alpha_idx: # one sweep in defined as N attempts of flip for each types
            energy_i = -(np.cos(np.repeat(self.alpha_config[idx][np.newaxis, :], 4, axis=0) - self.alpha_config[self.nbr[idx], :])).sum(axis=0)
            spin_temp = self.alpha_config[idx] + prop[idx]
            energy_f = -(np.cos(np.repeat(spin_temp[np.newaxis, :], 4, axis=0) - self.alpha_config[self.nbr[idx], :])).sum(axis=0)
            delta_e = energy_f - energy_i

            dec = acceptance[idx] < np.exp(-beta * delta_e)

            self.alpha_config[idx, dec] += prop[idx, dec]


    """ 
    calculate the energy of a given configuration  
    input: S/alpha configuration in list
             H/external field, defult 0
    """
    def get_energy(self):
        energy_ = np.zeros((self.N, self.alpha_config.shape[1]))

        for idx in range(0, self.N): # calculate energy per alpha and types
            energy_[idx] = -(np.cos((np.repeat(self.alpha_config[idx][np.newaxis, :], 4, axis=0)-self.alpha_config[self.nbr[idx], :]))).sum(axis=0) #nearst neighbor of kth alfa

        return energy_
        
    """
        Let the system evolve to equilibrium state
    """
    def equilibrate(self, max_n_sweeps=int(1e4), temperature=None, show = False):
        if temperature is not None:
            self.temperature = temperature
        dic_thermal_t = {'energy': []}
        beta = 1.0/self.temperature
        energy_temp = 0
        alphas = []
        for k in list(range(max_n_sweeps)):
            self.sweep()
            energy = np.sum(self.get_energy())/self.N/2
            dic_thermal_t['energy'] += [energy]
            alphas.append(self.alpha_config)
            if show & (k%1e3 ==0):
                print(f'sweeps={k+1}')
                print(f'energy={energy}')
                self.show()
                self.show_map(text='Start equilibrate')
            if ((abs(energy-energy_temp) / abs(energy)<1e-4) & (k>500)) or k == max_n_sweeps-1:
                print(f'\nequilibrium state is reached at T={self.temperature}')
                print(f'#sweep={k}')
                print(f'energy={energy}')
                if show:
                    self.show()
                    self.show_map(text='End equilibrate')
                break
            energy_temp = energy
        nstates = len(dic_thermal_t['energy'])
        energy = np.average(dic_thermal_t['energy'][int(nstates/2):])
        self.energy = energy
        energy2 = np.average(np.power(dic_thermal_t['energy'][int(nstates/2):], 2))
        self.Cv = (energy2-energy**2)*beta**2

        return self._inv_tranf()

    """
        Removing multiple alphas over (-pi, pi)
    """
    def get_alphas(self, degree=False):
        x = np.cos(self.alpha_config)
        y = np.sin(self.alpha_config)
        alphas_norm = np.arctan2(y, x)
        return alphas_norm

    def _transf(self, t, c, d, a, b):
        return c + ((d-c)/(b-a)) * (t - a)

    def _inv_tranf(self):
        return self._transf(self.get_alphas(), 0, self.support_end, -pi, pi)


    """
    To see thermo quantities evolve as we cooling the systems down
    input: T_inital: initial tempreature
           T_final: final temperature
           sample/'log' or 'lin',mean linear sampled T or log sampled( centered at critical point)
    """
    def annealing(self, t_init=2.5, t_final=0.1, nsteps= 20, show_equi=False):
        # initialize alphas. Orientations are taken from 0 - 2pi randomly.
        # initialize alphas configuration
        dic_thermal = {
            'temperature': list(np.linspace(t_init, t_final, nsteps)),
            'energy': [],
            'Cv': []
        }
        for T in dic_thermal['temperature']:
            self.equilibrate(temperature=T)
            if show_equi:
                self.show_map()
            dic_thermal['energy'] += [self.energy]
            dic_thermal['Cv'] += [self.Cv]
        plt.plot(dic_thermal['temperature'], dic_thermal['Cv'],'.')
        plt.ylabel(r'$C_v$')
        plt.xlabel('T')
        plt.show()
        plt.plot(dic_thermal['temperature'], dic_thermal['energy'],'.')
        plt.ylabel(r'$\langle E \rangle$')
        plt.xlabel('T')
        plt.show()
        return dic_thermal

    """
    convert configuration inz list to matrix form
    """
    @staticmethod
    def list2matrix(s):
        n = int(np.size(s))
        l = int(np.sqrt(n))
        s = np.reshape(s, (l, l))
        return s

    """
    visualize a configuration
    input：S/ alpha configuration in list form
    """
    def show(self, text = None):
        for i in range(self.alpha_config.shape[1]):
            config_matrix = self.list2matrix(self.alpha_config[:, i])
            x, y = np.meshgrid(np.arange(0, self.L), np.arange(0, self.L))
            u = np.cos(config_matrix)
            v = np.sin(config_matrix)
            plt.figure(figsize=(10, 10))
            Q = plt.quiver(x, y, u, v, self.list2matrix(self._inv_tranf()[:, i]))
            #plt.quiverkey(Q, 0.1, 0.1, 1, r'$alpha$', labelpos='E',  coordinates='figure')
            plt.colorbar()
            plt.title(f'T={self.temperature}, #alphas={str(self.L)} {str(self.L)} {text}')
            plt.axis('off')
        #plt.savefig(f'{random.randint(1,100)}.png')

    def show_map(self, text=""):
        for i in range(self.alpha_config.shape[1]):
            fig = plt.figure(figsize=(20, 10))
            ax0 = fig.add_subplot(1, 3, 1)
            ax0.set_title(f"T={self.temperature}, {text}", fontsize=25)
            im0 = ax0.imshow(self.list2matrix(self._inv_tranf()[:, i]),  vmin=0, vmax=self.support_end, cmap=cm.seismic)
            divider0 = make_axes_locatable(ax0)
            cax0 = divider0.append_axes("right", size="10%", pad=0.05)

            fig.colorbar(im0, cax=cax0)
            #plt.savefig(f'{random.randint(1,100)}.png')
