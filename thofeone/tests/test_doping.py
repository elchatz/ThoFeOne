''' Check resuts remain the same with varying mesh size
    Mesh is constant
    Using the default physical constants from constants.py
'''
import constants as const
from constants import DefaultProblem as cdp
import thofeone as tf
import discretized_quantum as dq
import discretized_poisson as dp
import sc_problem as scp
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
n_iter = 1000
w2deg = 100

def prepare(w2deg, mesh):
    ''' Build and solve the self-consistent problem
        w2deg:    Width of the 2DEG
        The stack is from the paper 'Unveiling the charge distribution 
        of a GaAs-based nanoelectronic device'
    '''

    gaas_data_0 = {'layer_name':'GaAs', 'size':w2deg, 'mesh_spacing':mesh, 'U':cdp.Ec_gaas, 
                   'mass':cdp.m_gaas, 'nd':0, 'epsilon':cdp.epsilon_GaAs}
    algaas_data_1 = {'layer_name':'AlGaAs', 'size':d1, 'mesh_spacing':mesh, 'U':cdp.Ec_algaas, 
                     'mass':cdp.m_algaas, 'nd':0, 'epsilon':cdp.epsilon_AlGaAs}
    algaas_data_2 = {'layer_name':'AlGaAs', 'size':d2, 'mesh_spacing':mesh, 'U':cdp.Ec_algaas, 
                     'mass':cdp.m_algaas, 'nd':n_d, 'epsilon':cdp.epsilon_AlGaAs}
    algaas_data_3 = {'layer_name':'AlGaAs', 'size':d3, 'mesh_spacing':mesh, 'U':cdp.Ec_algaas, 
                     'mass':cdp.m_algaas, 'nd':0, 'epsilon':cdp.epsilon_AlGaAs}
    gaas_data_1 = {'layer_name':'GaAs', 'size':d4, 'mesh_spacing':mesh, 'U':cdp.Ec_gaas, 
                   'mass':cdp.m_gaas, 'nd':0, 'epsilon':cdp.epsilon_GaAs}
    
    layers_data = [gaas_data_0, algaas_data_1, algaas_data_2, algaas_data_3, gaas_data_1]
    
    return layers_data

def hetero(aa):
    layers_data = prepare(w2deg = w2deg, mesh = 1)
    qp = dq.DiscretizedQuantum(aa = aa)
    Diag, Upper = qp.build_system(layers_data = layers_data)
    Es, Psis = qp.eigs_quantum()    
    scp.init_ILDOS(qp, layers_data, qp.U)
    return -((1*np.sum(qp.nd)/qp.pois_conversion)*1e21)/d2

@pytest.fixture
def hetero_1nm():
    return hetero(1).astype(float)

@pytest.fixture
def hetero_05nm():
    return hetero(0.5).astype(float)

def test_mesh_doping(hetero_1nm, hetero_05nm):
    assert np.isclose(hetero_1nm, hetero_05nm)
