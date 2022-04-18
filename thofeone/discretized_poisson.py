import math
import numpy as np
import scipy
import scipy.sparse.linalg
from scipy.sparse import csc_matrix

from constants import *

class DiscretizedPoisson:
    """ Build the Discretized Poisson Problem
    This class will need to be turned into a version with variable mesh
    if you need to solve the variable mesh discretized self-consistent problem
    """
    
    def __init__(self):
        ''' 
        Initialize the descritized Poisson problem     
        '''
        # Capacitance matrix: Discretized version of the Laplacian: CU = q
        self.cappa = np.array([], dtype=float)   
        self.positions = np.array([], dtype=float) # x positions [nm]
        self.n_cells = 0 # Number of cells for Poisson [integer]
        self.pois_conversion = ee/(aa*nm*epsilon_0) 
        
    def fill_capa(self, positions, gates, epsilon): 
        """ Fill in the capacitance matrix
        positions:  Array to hold mesh positions
        gates:      2-tuple of bool. Shows whether there is a Schottky gate (True)
                    or not (False). First element is for first elements of array,
                    second for last element of array. 
                    (Top or bottom of array????)
        epsilon:    Relative dielectric constant
        """
        self.positions = positions
        self.n_cells = np.shape(positions)[0]
        n_cells = self.n_cells
        
        I =  np.zeros(4*n_cells + 10)
        J =  np.zeros(4*n_cells + 10)
        K =  np.zeros(4*n_cells + 10)              
        n_C_matrix = 0                  
        
        # Matrix elements counter for the Capacitance matrix
        # I,J are the line and column indices
        # K is the value in the matrix

        for i in np.arange(2,n_cells-1,1):
            ci = epsilon[i]/(self.positions[i]-self.positions[i-1])
            di = epsilon[i]/(self.positions[i+1]-self.positions[i])
            I[n_C_matrix] = i
            J[n_C_matrix] = i
            K[n_C_matrix] = -ci
            n_C_matrix += 1
            I[n_C_matrix] = i
            J[n_C_matrix] = i-1
            K[n_C_matrix] = +ci
            n_C_matrix += 1

        for i in np.arange(1,n_cells-2,1):
            ci = epsilon[i]/(self.positions[i]-self.positions[i-1])
            di = epsilon[i]/(self.positions[i+1]-self.positions[i])
            I[n_C_matrix] = i
            J[n_C_matrix] = i
            K[n_C_matrix] = -di
            n_C_matrix += 1
            I[n_C_matrix] = i
            J[n_C_matrix] = i+1
            K[n_C_matrix] = +di
            n_C_matrix += 1

        d0 = epsilon[0]/(self.positions[1]-self.positions[0])
        c1 = epsilon[1]/(self.positions[1]-self.positions[0])

        # Begin filling for boundry conditions 

        # If there is a metallic gate at the bottom
        if gates[0]:
            I[n_C_matrix] = 0
            J[n_C_matrix] = 0
            K[n_C_matrix] = -1/d0
            n_C_matrix += 1
            I[n_C_matrix] = 0
            J[n_C_matrix] = 1
            K[n_C_matrix] = 1
            n_C_matrix += 1
            I[n_C_matrix] = 1
            J[n_C_matrix] = 0
            K[n_C_matrix] = -c1/d0 
            n_C_matrix += 1
        else:
            I[n_C_matrix] = 0
            J[n_C_matrix] = 0
            K[n_C_matrix] = -d0
            n_C_matrix += 1
            I[n_C_matrix] = 1
            J[n_C_matrix] = 0
            K[n_C_matrix] = d0
            n_C_matrix += 1
            I[n_C_matrix] = 1
            J[n_C_matrix] = 1
            K[n_C_matrix] = -c1
            n_C_matrix += 1
            I[n_C_matrix] = 0
            J[n_C_matrix] = 1
            K[n_C_matrix] = c1
            n_C_matrix += 1

        dNM = epsilon[n_cells-2]/(self.positions[n_cells-1] - self.positions[n_cells-2])
        cN = epsilon[n_cells-1]/(self.positions[n_cells-1] - self.positions[n_cells-2])

        # If there is a metallic gate at the top
        if gates[1]:
            I[n_C_matrix] = n_cells-1
            J[n_C_matrix] = n_cells-1
            K[n_C_matrix] = -1/cN
            n_C_matrix += 1
            I[n_C_matrix] = n_cells-1
            J[n_C_matrix] = n_cells-2
            K[n_C_matrix] = 1
            n_C_matrix += 1
            I[n_C_matrix] = n_cells-2
            J[n_C_matrix] = n_cells-1
            K[n_C_matrix] = -cN/dNM
            n_C_matrix += 1
        else:
            I[n_C_matrix] = n_cells-1
            J[n_C_matrix] = n_cells-1
            K[n_C_matrix] = -cN
            n_C_matrix += 1
            I[n_C_matrix] = n_cells-1
            J[n_C_matrix] = n_cells-2
            K[n_C_matrix] = cN
            n_C_matrix += 1
            I[n_C_matrix] = n_cells-2
            J[n_C_matrix] = n_cells-2
            K[n_C_matrix] = -dNM
            n_C_matrix += 1
            I[n_C_matrix] = n_cells-2
            J[n_C_matrix] = n_cells-1
            K[n_C_matrix] = dNM
            n_C_matrix += 1

        J = J.astype(int)
        I = I.astype(int)

        self.capa = csc_matrix((K[0:n_C_matrix], (I[0:n_C_matrix], J[0:n_C_matrix])))

        return self.capa
    
    
    def solve_pb(self, qp, gates, V_bot = 0, V_top = 0):
        q_quant = qp.ildos.copy()
        q_quant = q_quant*self.pois_conversion
        q_pois = (q_quant + qp.nd).copy()
        if gates[0]:
            q_pois[0] =  V_bot
        if gates[1]:
            q_pois[-1] = V_top
        U_pois = scipy.sparse.linalg.spsolve(self.capa, q_pois)
        return U_pois
    
