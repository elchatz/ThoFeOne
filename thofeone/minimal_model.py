from constants import *

import numpy as np
import matplotlib.pyplot as plt


""" nd and d1, d2 etc should be given in in /m2
    res in /m2
"""

def ns_0D(rho, Ef):
    return rho*Ef

def ns_7(rho, epsilon = 12, Vs = 0, d1 = 0, d2 = 0, d3 = 0, d4 = 0, nd = 1e15):
    '''
    Eq. 7 : Cap layer not doped
    '''
    dtot = d1 + d2 + d3 + d4
    res = (-Vs + ((((d2/2)+d3+d4)/(epsilon*epsilon_0))*-1*ee*nd)) / (((1/(-1*ee*rho)) + 
                                                                       (dtot/(epsilon*epsilon_0))) * -1*ee)
    res = np.where(res>0, res, 0)
    return res
    
def ns_7_poisson(epsilon = 12, Vs = 0, d1 = 0, d2 = 0, d3 = 0, d4 = 0, nd = 1e15):
    '''
    Eq. 7 : No quantum part
    '''
    dtot = d1 + d2 + d3 + d4
    res = (-Vs + ((((d2/2)+d3+d4)/(epsilon*epsilon_0))*-1*ee*nd)) / ((dtot/(epsilon*epsilon_0)) * -1*ee)
    res = (np.where(res>0, res, 0))
    return res

def ns_7_poisson_2(epsilon = 12, Vs = 0, d1 = 0, d2 = 0, d3 = 0, d4 = 0, nd = 1e15):
    '''
    Eq. 7 : No quantum part
    '''
    dtot = d1 + d2 + d3 + d4
    res = -(epsilon*epsilon_0*Vs)/(-1*ee*dtot) + (((d2/2)+d3+d4)/dtot)*nd
    res = (np.where(res>0, res, 0))
    return res
    
def ns_7_correction(rho, z2deg_arr, epsilon = 12, Vs = 0, 
                    d1 = 0, d2 = 0, d3 = 0, d4 = 0, nd = 1e15):
    '''
    Eq. 7 : Correction for the effective location of the 2DEG, quantum-poisson 
    '''
    for z in z2deg_arr:
        if np.isnan(z):
            dtot = d1 + d2 + d3 + d4
            res = (-Vs + ((((d2/2)+d3+d4)/(epsilon*epsilon_0))*-1*ee*nd)) / (((1/(-1*ee*rho)) + 
                                                                              (dtot/(epsilon*epsilon_0))) * -1*ee)
        else:
            dtot = d1 + d2 + d3 + d4 + z
            res = (-Vs + ((((d2/2)+d3+d4)/(epsilon*epsilon_0))*-1*ee*nd)) / (((1/(-1*ee*rho)) + 
                                                                               (dtot/(epsilon*epsilon_0))) * -1*ee)
    res = (np.where(res>0, res, 0))
    return res

def ns_7_p_correction(z2deg_arr, epsilon = 12, Vs = 0, d1 = 0, d2 = 0, d3 = 0, d4 = 0, nd = 1e15):
    '''
    Eq. 7 : Correction for the effective location of the 2DEG, no quantum 
    '''
    for z in z2deg_arr:
        if np.isnan(z):
            dtot = d1 + d2 + d3 + d4
            res = (-Vs + ((((d2/2)+d3+d4)/(epsilon*epsilon_0))*-1*ee*nd)) / ((dtot/(epsilon*epsilon_0)) * -1*ee)
        else:
            dtot = d1 + d2 + d3 + d4 + z
            res = (-Vs + ((((d2/2)+d3+d4)/(epsilon*epsilon_0))*-1*ee*nd)) / ((dtot/(epsilon*epsilon_0)) * -1*ee)
    res = (np.where(res>0, res, 0))
    return res
    
def Vg_7(epsilon = 12, ns = 1e15, d1 = 0, d2 = 0, d3 = 0, d4 = 0, nd = 1e15):
    '''
    Eq. 7: Quantum-Poisson, solving for Vg
    '''
    dtot = d1 + d2 + d3 + d4
    res =  -((1/(-1*ee*rho)) + ((dtot)/epsilon))*(-1*ee*ns) + (((d2/2)+d3+d4)/(epsilon*epsilon_0))*-1*ee*nd
    return res

def Vg_7_poisson(rho, epsilon = 12, ns = 1e15, d1 = 0, d2 = 0, d3 = 0, d4 = 0, nd = 1e15):
    '''
    Eq. 7: Poisson only, solving for Vg
    '''
    dtot = d1 + d2 + d3 + d4
    res =  -(((dtot)/epsilon))*(-1*ee*ns) + (((d2/2)+d3+d4)/(epsilon*epsilon_0))*-1*ee*nd
    return res

def ns_20(rho, epsilon = 12, Vs = 0, d1 = 0, d2 = 0, d4 = 0, nd = 1e15):
    '''
    Eq. 20 : Cap layer doped
    '''
    dtot = d1 + d2 + d4
    res = ( (Vs/ee) + (((d2/2)+(d4/2))/(epsilon*epsilon_0))*nd) / (((1/((-1*ee)**2*rho)) + 
                                                                     (dtot/(epsilon*epsilon_0))))
    res = (np.where(res>0, res, 0))
    return res

def ns_20_correction(rho, z2deg_arr, epsilon = 12, Vs = 0, d1 = 0, d2 = 0, d4 = 0, nd = 1e15):
    '''
    Eq. 20 : Correction for the location of the GaAs/AlGaAs interface (charge centroid)
    '''
    for z in z2deg_arr:
        if np.isnan(z):
            dtot = d1 + d2 + d4
            res = ( (Vs/ee) + (((d2/2)+(d4/2))/(epsilon*epsilon_0))*nd) / (((1/((-1*ee)*rho)) + 
                                                                             (dtot/(epsilon*epsilon_0))))
        else:
            dtot = d1 + d2 + d4 + z
            res = ( (Vs/ee) + (((d2/2)+(d4/2))/(epsilon*epsilon_0))*nd) / (((1/((-1*ee)*rho)) + 
                                                                             (dtot/(epsilon*epsilon_0))))
    res = (np.where(res>0, res, 0))
    return res

def ns_PESCA_20(epsilon = 12, Vs = 0, d1 = 0, d2 = 0, d4 = 0, nd = 1e15):
    '''
    Eq. 20 PESCA
    '''
    dtot = d1 + d2 + d4
    res = ( (Vs/ee) + (((d2/2)+(d4/2))/(epsilon*epsilon_0))*nd) / ((dtot/(epsilonepsilon_0)))
    res = (np.where(res>0, res, 0))
    return res

def ns_PESCA_20_correction(z2deg_arr, epsilon = 12, Vs = 0, d1 = 0, d2 = 0, d4 = 0, nd = 1e15):
    for z in z2deg_arr:
        if np.isnan(z):
            dtot = d1 + d2 + d4
            res = ( (Vs/ee) + (((d2/2)+(d4/2))/(epsilon*epsilon_0))*nd) / ((dtot/(epsilon*epsilon_0)))
        else:
            dtot = d1 + d2 + d4 + z
            res = ( (Vs/ee) + (((d2/2)+(d4/2))/(epsilon*epsilon_0))*nd) / ((dtot/(epsilon*epsilon_0)))
    res = (np.where(res>0, res, 0))
    return res

def Vg_PESCA_20(rho, epsilon = 12, ns = 1e15, d1 = 0, d2 = 0, d4 = 0, nd = 1e15):
    dtot = d1 + d2 + d4
    res =  -(((dtot)/epsilon)*epsilon_0)*(-1*ee*ns) + (((d2/2)+(d4/2))/(epsilon*epsilon_0))*-1*ee*nd
    return res

def Vg_20(rho, epsilon = 12, ns = 1e15, d1 = 0, d2 = 0, d4 = 0, nd = 1e15):
    dtot = d1 + d2 + d4
    res =  (-((1/(-1*ee*rho)) + ((dtot)/epsilon))*(-1*ee*ns)) + (((d2/2)+(d4/2)/
                                                                  (epsilon*epsilon_0))*-1*ee*nd)
    return res