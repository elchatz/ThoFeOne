'''Routines for self-consistent problem
'''
from constants import *
from constants import DefaultProblem as cdp
import numpy as np

def init_ILDOS(qp, layers_data, U):
    init_en = 0
    qp.update_U(U.copy())
    Es, Psis = qp.eigs_quantum()
    qp.solve_ildos(0, init = True)
    q_quant = qp.ildos.copy()
    q_quant = q_quant*rho_2DEG_aa(qp.aa, qp.mass)
    qp.ildos = q_quant.copy()
    return

def build_pois(qp, pp, gates, layers_data):
    n_cells = qp.meshpoints_total   
    # Fill the capacitance matrix
    C = pp.fill_capa(positions = qp.positions, gates = gates, epsilon = qp.epsilon) 
    #U_pois = np.zeros(np.shape(C)[0])
    return