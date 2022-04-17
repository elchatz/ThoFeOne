''' Defining physical constants
'''
import numpy as np

ee = 1.60217662e-19 # Electron charge [Coulomb]
epsilon_0 = 8.8541878128e-12 # Vacuum permitivity [F/m]
nm = 1e-9 # nm [m]
hbar = 1.0545718e-34 # Reduced Planck constant [J⋅s / m2⋅kg⋅s]
m_e  = 9.10938356e-31 # Electron mass [kg]

def rho_2DEG_aa(m, aa):
    ''' Returns the 2DEG density in units of [1/J]
        m:    Effective mass [m_e]
        aa:   Hamiltonian discretization constant [nm]
    '''
    return ((aa*nm)**2*ee)*m*m_e/(hbar**2*np.pi)

def rho_2DEG_nm(m):
    ''' Returns the 2DEG density in units of [1/J]
        m:    Effective mass [m_e]
    '''
    return ((nm)**2*ee)*m*m_e/(hbar**2*np.pi)

def rho_2DEG_SI(m):
    ''' Returns the 2DEG density in units of 1/(J⋅m2)
        m:    Effective mass [m_e]
    '''
    return (m*m_e*ee)/(hbar**2*np.pi)

def rho_2DEG(m):
    ''' Returns the 2DEG density in units of 1/(ee⋅J⋅m2) or 1/(eV⋅m2)
    '''
    res = (m*m_e)/(hbar**2*np.pi)
    return res
    
class DefaultProblem:
    ''' Contains values for a default self-consistent problem
    '''
    # Default physical constants
    epsilon_GaAs = 12.93 # Relative dielectric constant of GaAs
    epsilon_AlGaAs = 11.93 # Relative dielectric constant of AlGaAs
    m_gaas = 0.067 # Effective mass for GaAs
    m_algaas = 0.091 # Effective mass for AlGaAs
    Ec_gaas = 0 # Conduction band for GaAs (Fermi level) [eV]
    Ec_algaas = 0.292 #eV
    
    # Default values for solution
    aa = 1 # Discretization constant for Hamiltonian [nm]
    grid_size = 1 # Mesh spacing [nm]
    #t = hbar**2/(2*m_e*(aa*nm)**2) # Hopping elements
    #eV_2_t = ee/t # Converting energy units to t
    # Constant for conversion during self-consistent calculations
    #pois_conversion = ee/(aa*epsilon_0)  
        
