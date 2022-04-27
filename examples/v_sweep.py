''' Example of a simple heterostructure with constant mesh
    Using the default physical constants from constants.py
'''
import constants as const
from constants import DefaultProblem as dpr
import thofeone as tf
import discretized_quantum as dq
import discretized_poisson as dp
from minimal_model import *
import sc_problem as scp


import matplotlib.pyplot as plt
import numpy as np

def prepare(w2deg, n_d):
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
    
    return layers_data

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
    n_iter = 2000
    ns_arr = []
    V_arr = np.arange(-0.8, 0.0, 0.05)
    
    layers_data = prepare(w2deg, n_d)
    qp = dq.DiscretizedQuantum()
    Diag, Upper = qp.build_system(layers_data = layers_data)
    Es, Psis = qp.eigs_quantum()
    
    # Preparing vqlues for minimal model
    nd_min = n_d*d2 # Doping in /nm3 and d2 in nm
    nd_min = nd_min*1e18 # Convert to /m2 for analytical model solution
    
    rho = const.rho_2DEG_SI(dpr.m_gaas)
    nsvsVsc = UndopedCap.ns(rho = rho, d1 = d1*const.nm, d2 = d2*const.nm, d3=d3*const.nm, 
                     d4 = d4*const.nm, nd = nd_min, Vs = V_arr-barrier_height)
    nsvsVsc_p = UndopedCap.ns_poisson(d1 = d1*const.nm, d2 = d2*const.nm, d3=d3*const.nm, 
                               d4 = d4*const.nm, nd = nd_min, Vs = V_arr-barrier_height)

    # Sweep gate voltage
    print('---- Start of gate sweep ----')
    for Vg in V_arr:
        print('Vg = ', np.round(Vg, decimals = 2), 'V')
        scp.init_ILDOS(qp, layers_data, qp.U)
        gates = (False, True)
        pp = dp.DiscretizedPoisson()
        scp.build_pois(qp, pp, gates, layers_data)
        qp.simple_mixing(pp, gates = gates, mu = 0, n_iter = n_iter, V_top = Vg-barrier_height)
        ns = np.sum(qp.ildos[qp.positions < (w2deg+dpr.grid_size)])
        ns_arr = np.append(ns_arr, ns)
    print('---- End of gate sweep ----')
    
    # Plot results
    fig, ax = plt.subplots(figsize=(9,5))
    ax.plot(V_arr, ns_arr*1e18/1e15, marker='o', mfc=None, alpha=1, linewidth=0, label = '1D simulation' )
    ax.plot(V_arr, nsvsVsc/1e15, alpha=1, linewidth=3.0, label = 'Quantum capacitance')
    ax.plot(V_arr, nsvsVsc_p/1e15, alpha=1, linewidth=3.0, label = 'Poisson only')
    
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                 ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(15)
    
    plt.legend(fontsize=15)
    plt.xlim(-0.5, 0.0)
    plt.xlabel(r'$V_g$ (V)')
    plt.ylabel(r'$n_g$ ($10^{15} m^{-2}$)') 
    plt.grid()
    plt.savefig('analytical_simulation.png')
    
    


