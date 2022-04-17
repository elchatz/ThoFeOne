''' Example of a simple heterostructure with constant mesh
    Using the default physical constants from constants.py
'''
import constants as const
from constants import DefaultProblem as dpr
import thofeone as tf
import discretized_quantum as dq
import discretized_poisson as dp
import sc_problem as scp
import matplotlib.pyplot as plt
import numpy as np

def build_solve(w2deg, n_d):
    ''' Build and solve the self-consistent problem
        w2deg:    Width of the 2DEG
        The stack is from the paper 'Unveiling the charge distribution 
        of a GaAs-based nanoelectronic device'
    '''

    gaas_data_0 = {'layer_name':'GaAs', 'size':w2deg, 'mesh_spacing':dpr.grid_size, 'U':dpr.Ec_gaas, 
                   'mass':dpr.m_gaas, 'nd':0, 'epsilon':dpr.epsilon_GaAs}
    algaas_data_1 = {'layer_name':'AlGaAs', 'size':d1, 'mesh_spacing':dpr.grid_size, 'U':dpr.Ec_algaas, 
                     'mass':dpr.m_algaas, 'nd':0, 'epsilon':dpr.epsilon_AlGaAs}
    algaas_data_2 = {'layer_name':'AlGaAs', 'size':d2, 'mesh_spacing':dpr.grid_size, 'U':dpr.Ec_algaas, 
                     'mass':dpr.m_algaas, 'nd':n_d, 'epsilon':dpr.epsilon_AlGaAs}
    algaas_data_3 = {'layer_name':'AlGaAs', 'size':d3, 'mesh_spacing':dpr.grid_size, 'U':dpr.Ec_algaas, 
                     'mass':dpr.m_algaas, 'nd':0, 'epsilon':dpr.epsilon_AlGaAs}
    gaas_data_1 = {'layer_name':'GaAs', 'size':d4, 'mesh_spacing':dpr.grid_size, 'U':dpr.Ec_gaas, 
                   'mass':dpr.m_gaas, 'nd':0, 'epsilon':dpr.epsilon_GaAs}
    
    layers_data = [gaas_data_0, algaas_data_1, algaas_data_2, algaas_data_3, gaas_data_1]

    qp = dq.DiscretizedQuantum(aa = aa)
    Diag, Upper = qp.build_system(layers_data = layers_data)
    Es, Psis = qp.eigs_quantum()
    
    return qp, layers_data, qp.U

if __name__ == '__main__':
    
    # Heterostructure-specific variables
    # Doping value in [cm-3], conversion to [/nm3], equal to the value required
    # by the simulator. Inside the simulator, this value is multiplied by the
    # mesh grid width in nm so that each mesh point has nd [/nm2]
    n_d = 2.34353747938051e+17*1e-21 
    barrier_height = 0.75 # Schottky barrier height [eV]
    # Layer widths in [nm]
    d1 = 25
    d2 = 65
    d3 = 10
    d4 = 10
    dtot = d1 + d2 + d3 + d4
    w2deg = 100
    aa = dpr.aa
    
    # Build the quantum problem and solve once
    qp, layers_data, U_quant = build_solve(w2deg, n_d)
    scp.init_ILDOS(qp, layers_data, U_quant)
    
    # Doping density for mesh spacing = 1 nm
    # If changing the spacing, you need to take this into account in the sum in 
    # order to calculate the derivative. Probably include an integration routine
    # inside discretized_quantum. Sizes for each mesh position are not stored, need
    # to be calculated
    print('Doping density', -((1*np.sum(qp.nd)/qp.pois_conversion)*1e21)/d2, '/cm2')
    
    # Build Poisson problem
    gates = (False, True)
    pp = dp.DiscretizedPoisson(aa = aa)
    scp.build_pois(qp, pp, gates, layers_data)
    
    # Solve self-consistent at gate voltage = -0.3 V
    Vg = -0.3
    n_iter = 1000
    qp.simple_mixing(pp, gates = gates, mu = 0, n_iter = n_iter, V_top = Vg-barrier_height)
    
    fig, ax = plt.subplots()
    ax.plot(qp.positions, (qp.Eb/qp.eV_2_t) + qp.U/qp.eV_2_t, 'g')
    plt.xlabel('position (nm)')
    plt.ylabel(r'$E_c$ + $U_{pois}$ [eV]')
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + 
                 ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(15)
    plt.grid(b=True)
    plt.tight_layout()
    plt.savefig('band.png')
    
    # This prints out the charge density in /m3 for a CONSTANT mesh 
    # spacing = discretization constant for Hamiltonian. For variable mesh, 
    # we need the mesh at each location (L array), or just print in /nm2
    fig, ax = plt.subplots()
    ax.plot(qp.positions, (qp.ildos*1e18)/(qp.aa*const.nm), 'g')
    plt.xlabel('position (nm)')
    plt.ylabel(r'Charge ($m^{-3}$)')
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + 
                 ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(15)
    plt.grid(b=True)
    plt.savefig('ILDOS.png')
    


