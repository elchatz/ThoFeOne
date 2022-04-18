''' Check resuts remain the same with varying mesh size
    Mesh is constant
    Using the default physical constants from constants.py
'''
import constants as const
import sc_problem as scp
from constants import DefaultProblem as dpr
import thofeone as tf
import discretized_quantum as dq
import discretized_poisson as dp
from minimal_model import *

import numpy as np
import pytest

n_d = 2.34353747938051e+17*1e-21 
barrier_height = 0.75 
surf_charge = 6.135378601385091e+18*1e-21 
d1 = 25
d2 = 65
d3 = 10
d4 = 10
dtot = d1 + d2 + d3 + d4
Vg = -0.3
w2deg = 100

def prepare(w2deg, mesh):
    ''' Build and solve the self-consistent problem
        w2deg:    Width of the 2DEG
        The stack is from the paper 'Unveiling the charge distribution 
        of a GaAs-based nanoelectronic device'
    '''

    gaas_data_0 = {'layer_name':'GaAs', 'size':w2deg, 'mesh_spacing':mesh, 'U':dpr.Ec_gaas, 
                   'mass':dpr.m_gaas, 'nd':0, 'epsilon':dpr.epsilon_GaAs}
    algaas_data_1 = {'layer_name':'AlGaAs', 'size':d1, 'mesh_spacing':mesh, 'U':dpr.Ec_algaas, 
                     'mass':dpr.m_algaas, 'nd':0, 'epsilon':dpr.epsilon_AlGaAs}
    algaas_data_2 = {'layer_name':'AlGaAs', 'size':d2, 'mesh_spacing':mesh, 'U':dpr.Ec_algaas, 
                     'mass':dpr.m_algaas, 'nd':n_d, 'epsilon':dpr.epsilon_AlGaAs}
    algaas_data_3 = {'layer_name':'AlGaAs', 'size':d3, 'mesh_spacing':mesh, 'U':dpr.Ec_algaas, 
                     'mass':dpr.m_algaas, 'nd':0, 'epsilon':dpr.epsilon_AlGaAs}
    gaas_data_1 = {'layer_name':'GaAs', 'size':d4, 'mesh_spacing':mesh, 'U':dpr.Ec_gaas, 
                   'mass':dpr.m_gaas, 'nd':0, 'epsilon':dpr.epsilon_GaAs}
    
    layers_data = [gaas_data_0, algaas_data_1, algaas_data_2, algaas_data_3, gaas_data_1]
    
    return layers_data

@pytest.fixture
def sim():
    layers_data = prepare(w2deg = w2deg, mesh = 1)
    qp = dq.DiscretizedQuantum()
    Diag, Upper = qp.build_system(layers_data = layers_data)
    Es, Psis = qp.eigs_quantum()    
    scp.init_ILDOS(qp, layers_data, qp.U)
    gates = (False, True)
    pp = dp.DiscretizedPoisson()
    scp.build_pois(qp, pp, gates, layers_data)
    n_iter = 1500
    qp.simple_mixing(pp, gates = gates, mu = 0, n_iter = n_iter, V_top = Vg-barrier_height)
    return np.sum(qp.ildos)

@pytest.fixture
def min_mod():
    nd_min = n_d*d2
    nd_min = nd_min*1e18
    rho = const.rho_2DEG_SI(dpr.m_gaas)
    nsvsVsc_7 = ns_7(rho = rho, d1 = d1*const.nm, d2 = d2*const.nm, d3=d3*const.nm, 
                     d4 = d4*const.nm, nd = nd_min, Vs = Vg-barrier_height)
    return float(nsvsVsc_7)/1e18

def test_edens(sim, min_mod):
    assert np.isclose(sim, min_mod, rtol=3e-4, atol=3e-4)
