import numpy as np
import scipy
import math
import numpy.linalg as la
from scipy.linalg import *
import copy
import matplotlib.pyplot as plt
from constants import *

class DiscretizedQuantum:
    """ Build the Discretized Quantum Problem 
    """
    def __init__(self):
        self.positions = np.array([], dtype=float) # x [nm]
        self.U = np.array([], dtype=float) # [t]
        self.L = np.array([], dtype=float) # Helper variable for variable mesh
        self.mass = np.array([], dtype=float) # Effective mass array
        self.nd = np.array([], dtype=float) # Doping density [/nm2]
        self.epsilon = np.array([], dtype=float) # Dielectric constant
        self.Upper = np.array([], dtype=float) # Upper diagonal part of the discretized (tridiagonal) Hamiltonian [t]
        self.Diag = np.array([], dtype=float) # Diagonal part of the discretized Hamiltonian [t]
        self.meshpoints_total = 0
        self.Es = np.array([], dtype=float)
        self.Psis = np.array([], dtype=float)
        self.ildos = np.array([], dtype=float)
        self.aa = aa
        self.t = hbar**2/(2*m_e*(self.aa*nm)**2) 
        self.eV_2_t = ee/self.t
        # Normally, pois_conversion should be in the Poisson object, but it is not
        # created yet, so each class has its own local copy
        self.pois_conversion = ee/(self.aa*nm*epsilon_0) 
        self.Diag_init = np.array([], dtype=float) # Diagonal resulting from the band 
        self.Eb = np.array([], dtype=float)

    def build_system(self, layers_data):
        """ Discretized quantum problem with uniform mesh
        """
        meshpoints_start = 0
        start = 0 # Last x position after adding each layer
        potential = 0 # Band edge for each layer

        print('---- Bottom of stack ----')
        for i in np.arange(0, len(layers_data)):
            for key, value in layers_data[i].items():
                if key == 'layer_name':
                    print('Added layer material: ', value)
                if key == 'size':
                    end = start + value
                if key == 'mesh_spacing':
                    mesh_spacing = value
                    meshpoints = (end - start)/value
                    meshpoints = int(math.ceil(meshpoints))
                if key == 'U':
                    potential = value*self.eV_2_t
                if key == 'mass':
                    mass = value
                if key == 'nd':
                    nd = -self.pois_conversion*value*mesh_spacing
                if key == 'epsilon':
                    epsilon = value
            
            self.positions = np.append(self.positions, [i for i in np.linspace(
                start+mesh_spacing, end+mesh_spacing, meshpoints, endpoint=False)])
            self.U = np.append(self.U, [potential for i in np.linspace(
                start+mesh_spacing, end+mesh_spacing, meshpoints, endpoint=False)]) 
            self.mass = np.append(self.mass, [mass for i in np.linspace(
                start+mesh_spacing, end+mesh_spacing, meshpoints, endpoint=False)])
            self.nd = np.append(self.nd, [nd for i in np.linspace(
                start+mesh_spacing, end+mesh_spacing, meshpoints, endpoint=False)])
            self.epsilon = np.append(self.epsilon, [epsilon for i in np.linspace(
                start+mesh_spacing, end+mesh_spacing, meshpoints, endpoint=False)])
            layers_data[i].update({'meshpoints': meshpoints, 'start': start, 'end': end}) 
            start = end
            meshpoints_start = meshpoints_start+meshpoints

        print('---- Start of stack ----')
        
        self.meshpoints_total = meshpoints_start
        self.L = np.zeros(self.meshpoints_total)

        for i in np.arange(1, self.meshpoints_total-1):
            self.L[i] = (self.positions[i+1] - self.positions[i-1])**0.5/2**(1/2)
    
        self.L[0] = ((self.positions[2] - self.positions[0])/2)**0.5
        self.L[-1] = ((self.positions[-1] - self.positions[-3])/2)**0.5
        self.Upper =  np.zeros(self.meshpoints_total-1)   
        self.Diag =  np.zeros(self.meshpoints_total)    
        
        for i in np.arange(0,self.meshpoints_total-1):
            h_ij = 2/(self.mass[i] + self.mass[i+1])/(self.positions[i+1] - self.positions[i])
            self.Diag[i] = self.Diag[i] + h_ij/self.L[i]/self.L[i]
            self.Diag[i+1] = self.Diag[i+1] + h_ij/self.L[i+1]/self.L[i+1]
            self.Upper[i] = self.Upper[i] - h_ij/self.L[i]/self.L[i+1]

        self.Diag[0] = self.Diag[1]/2
        self.Upper[0] = self.Upper[1]/2
        self.Diag[-1] = self.Diag[-2]/2
        self.Upper[-1] = self.Upper[-2]/2 
        
        self.Diag_init = (self.Diag + self.U).copy()
        self.Eb = self.U.copy()
        self.Diag = (self.Diag + self.U).copy()
    
        return self.Diag, self.Upper
    
    def eigs_quantum(self):
        """ TODO: For now normalization only works for uniform mesh
        Solve the Discretized Quantum Problem 
        The sparse Hamiltonian tridiagonal matrix H is built from Diag and Upper
        """
        
        self.Es, self.Psis = eigh_tridiagonal(self.Diag, self.Upper)
        
        for i in np.arange(0, self.meshpoints_total, 1):
            self.Psis[:,i] = self.Psis[:,i] / self.L
            
        return self.Es, self.Psis
    
    def update_U(self, U):
        """ 
        Update the potential energy in the quantum region
        """
        
        self.U = U.copy()*self.eV_2_t
        self.Diag = (self.Diag_init + self.U).copy()
        
        return 
    
    def solve_ildos(self, mu, subbands=4, init = False):
        """  
        Calculation of ILDOS for one energy at T = 0 K
        mu:  Energy at which to calculate the ildos (in units of t)
        init:    Whether to initialize the ildos for the object or not (initialize) 
        """         
        
        ildos = np.zeros(np.shape(self.Psis)[1])
        eigenen = self.Es[0:subbands].copy()
        
        for i in np.arange(np.shape(eigenen)[0]):
            if eigenen[i] < mu:
                ildos = ildos + (mu - eigenen[i])*self.Psis[:,i]**2
        
        #if init == True:
        self.ildos = ildos.copy()
        
        return self.ildos
    
    def solve_quantum(self, U_pois, mu):
        self.update_U(-U_pois.copy())
        self.Es, self.Psis = self.eigs_quantum()
        ildos_sc = self.solve_ildos(mu = mu*self.eV_2_t, init = True)
        self.ildos = ildos_sc * rho_2DEG_nm(self.mass)
        return
        
    def bare_iteration(self, pp, gates, n_iter = 2000, V_bot = 0, V_top = 0):
        pprobl = copy.deepcopy(pp)
        for i in np.arange(n_iter):
            U_pois = pprobl.solve_pb(self, gates, V_bot = V_bot, V_top = V_top)
            solve_quantum(U_pois.copy(), mu = 0)
        return
    
    def simple_mixing(self, pp, gates, mu = 0, n_iter = 2000, V_bot = 0, V_top = 0, debug = False):
        print('------------ Start of self-consistent ----------------')
        pp = copy.deepcopy(pp)
        if debug == True:
            f, (ax1, ax2) = plt.subplots(1, 2, sharey = True)
        for i in np.arange(n_iter):
            U_pois = pp.solve_pb(self, gates, V_bot = V_bot, V_top = V_top)
            n_old = np.sum(self.ildos)
            dst_old = self.ildos.copy()
            self.solve_quantum(U_pois.copy(), mu = mu)
            n_new = np.sum(self.ildos)
            lmbda = 0.01
            coef = 1.
            if n_old != 0 :
                coef = n_new/n_old
            if coef > 1:
                lmbda = lmbda/coef
            self.ildos = ((1-lmbda) * dst_old + lmbda * self.ildos).copy()
            if debug == True:
                    ax1.plot(self.positions, self.ildos, label= str(i+1) + ' iteration')
                    ax1.set_xlabel('positions')
                    ax1.set_ylabel('ILDOS')
                    ax2.plot(self.positions, U_pois, label= str(i+1) + ' iteration')
                    ax2.set_xlabel('positions')
                    ax2.set_ylabel('U')
        if debug == True:
            plt.legend()
            plt.savefig('debug_ite.png')
        print('------------ End of self-consistent ----------------')
        return U_pois 

        
    
    
    
    
