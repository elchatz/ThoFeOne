import numpy as np
import matplotlib.pyplot as plt

class MinimalModel:
    """ nd and d1, d2 etc should be given in in m-2
        res in m-2
    """
    
    def __init__(self):
        self.m_e  = 9.10938356e-31
        self.epsilon_0 = 8.8541878128e-12 
        self.hbar = 1.0545718e-34
        self.nm = 1e-9
        self.ee = 1.60217662e-19

    def rho_2DEG_SI(self, m):
        return (self.m_e*m)/(self.hbar**2*np.pi)
    
    def ns_0D(self, rho, Ef):
        return rho*Ef
    
    def ns_7(self, rho, epsilon = 12, Vsc = 0, d1 = 0, d2 = 0, d3 = 0, d4 = 0, nd = 1e15):
        '''
        Eq. 7 : Cap layer not doped
        '''
        dtot = d1 + d2 + d3 + d4
        res = (-Vsc + ((((d2/2)+d3+d4)/(epsilon*self.epsilon_0))*-1*self.ee*nd)) / (((1/(-1*self.ee**2*rho)) + (dtot/(epsilon*self.epsilon_0))) * -1*self.ee)
        res = (np.where(res>0, res, 0))
        return res
        
    def ns_7_poisson(self, epsilon = 12, Vsc = 0, d1 = 0, d2 = 0, d3 = 0, d4 = 0, nd = 1e15):
        '''
        Eq. 7 : No quantum part
        '''
        dtot = d1 + d2 + d3 + d4
        res = (-Vsc + ((((d2/2)+d3+d4)/(epsilon*self.epsilon_0))*-1*self.ee*nd)) / ((dtot/(epsilon*self.epsilon_0)) * -1*self.ee)
        res = (np.where(res>0, res, 0))
        return res
    
    def ns_7_poisson_2(self, epsilon = 12, Vsc = 0, d1 = 0, d2 = 0, d3 = 0, d4 = 0, nd = 1e15):
        '''
        Eq. 7 : No quantum part
        '''
        dtot = d1 + d2 + d3 + d4
        res = -(epsilon*self.epsilon_0*Vsc)/(-1*self.ee*dtot) + (((d2/2)+d3+d4)/dtot)*nd
        res = (np.where(res>0, res, 0))
        return res
        
    def ns_7_correction(self, rho, z2deg_arr, epsilon = 12, Vsc = 0, d1 = 0, d2 = 0, d3 = 0, d4 = 0, nd = 1e15):
        '''
        Eq. 7 : Correction for the effective location of the 2DEG, quantum-poisson 
        '''
        for z in z2deg_arr:
            if np.isnan(z):
                dtot = d1 + d2 + d3 + d4
                res = (-Vsc + ((((d2/2)+d3+d4)/(epsilon*self.epsilon_0))*-1*self.ee*nd)) / (((1/(-1*self.ee**2*rho)) + (dtot/(epsilon*self.epsilon_0))) * -1*self.ee)
            else:
                dtot = d1 + d2 + d3 + d4 + z
                res = (-Vsc + ((((d2/2)+d3+d4)/(epsilon*self.epsilon_0))*-1*self.ee*nd)) / (((1/(-1*self.ee**2*rho)) + (dtot/(epsilon*self.epsilon_0))) * -1*self.ee)
        res = (np.where(res>0, res, 0))
        return res
    
    def ns_7_p_correction(self, z2deg_arr, epsilon = 12, Vsc = 0, d1 = 0, d2 = 0, d3 = 0, d4 = 0, nd = 1e15):
        '''
        Eq. 7 : Correction for the effective location of the 2DEG, no quantum 
        '''
        for z in z2deg_arr:
            if np.isnan(z):
                dtot = d1 + d2 + d3 + d4
                res = (-Vsc + ((((d2/2)+d3+d4)/(epsilon*self.epsilon_0))*-1*self.ee*nd)) / ((dtot/(epsilon*self.epsilon_0)) * -1*self.ee)
            else:
                dtot = d1 + d2 + d3 + d4 + z
                res = (-Vsc + ((((d2/2)+d3+d4)/(epsilon*self.epsilon_0))*-1*self.ee*nd)) / ((dtot/(epsilon*self.epsilon_0)) * -1*self.ee)
        res = (np.where(res>0, res, 0))
        return res
        
    def Vg_7(self, epsilon = 12, ns = 1e15, d1 = 0, d2 = 0, d3 = 0, d4 = 0, nd = 1e15):
        '''
        Eq. 7: Quantum-Poisson, solving for Vg
        '''
        dtot = d1 + d2 + d3 + d4
        res =  -((1/(-1*self.ee**2*rho)) + ((dtot)/epsilon))*(-1*self.ee*ns) + (((d2/2)+d3+d4)/(epsilon*self.epsilon_0))*-1*self.ee*nd
        return res
    
    def Vg_7_poisson(self, rho, epsilon = 12, ns = 1e15, d1 = 0, d2 = 0, d3 = 0, d4 = 0, nd = 1e15):
        '''
        Eq. 7: Poisson only, solving for Vg
        '''
        dtot = d1 + d2 + d3 + d4
        res =  -(((dtot)/epsilon))*(-1*self.ee*ns) + (((d2/2)+d3+d4)/(epsilon*self.epsilon_0))*-1*self.ee*nd
        return res
    
    def ns_20(self, rho, epsilon = 12, Vsc = 0, d1 = 0, d2 = 0, d4 = 0, nd = 1e15):
        '''
        Eq. 20 : Cap layer doped
        '''
        dtot = d1 + d2 + d4
        res = ( (Vsc/self.ee) + (((d2/2)+(d4/2))/(epsilon*self.epsilon_0))*nd) / (((1/((-1*self.ee)**2*rho)) + (dtot/(epsilon*self.epsilon_0))))
        res = (np.where(res>0, res, 0))
        return res

    def ns_20_correction(self, rho, z2deg_arr, epsilon = 12, Vsc = 0, d1 = 0, d2 = 0, d4 = 0, nd = 1e15):
        '''
        Eq. 20 : Correction for the location of the GaAs/AlGaAs interface (charge centroid)
        '''
        for z in z2deg_arr:
            if np.isnan(z):
                dtot = d1 + d2 + d4
                res = ( (Vsc/self.ee) + (((d2/2)+(d4/2))/(epsilon*self.epsilon_0))*nd) / (((1/((-1*self.ee)**2*rho)) + (dtot/(epsilon*self.epsilon_0))))
            else:
                dtot = d1 + d2 + d4 + z
                res = ( (Vsc/self.ee) + (((d2/2)+(d4/2))/(epsilon*self.epsilon_0))*nd) / (((1/((-1*self.ee)**2*rho)) + (dtot/(epsilon*self.epsilon_0))))
        res = (np.where(res>0, res, 0))
        return res

    def ns_PESCA_20(self, epsilon = 12, Vsc = 0, d1 = 0, d2 = 0, d4 = 0, nd = 1e15):
        '''
        Eq. 20 PESCA
        '''
        dtot = d1 + d2 + d4
        res = ( (Vsc/self.ee) + (((d2/2)+(d4/2))/(epsilon*self.epsilon_0))*nd) / ((dtot/(epsilon*self.epsilon_0)))
        res = (np.where(res>0, res, 0))
        return res
    
    def ns_PESCA_20_correction(self, z2deg_arr, epsilon = 12, Vsc = 0, d1 = 0, d2 = 0, d4 = 0, nd = 1e15):
        for z in z2deg_arr:
            if np.isnan(z):
                dtot = d1 + d2 + d4
                res = ( (Vsc/self.ee) + (((d2/2)+(d4/2))/(epsilon*self.epsilon_0))*nd) / ((dtot/(epsilon*self.epsilon_0)))
            else:
                dtot = d1 + d2 + d4 + z
                res = ( (Vsc/self.ee) + (((d2/2)+(d4/2))/(epsilon*self.epsilon_0))*nd) / ((dtot/(epsilon*self.epsilon_0)))
        res = (np.where(res>0, res, 0))
        return res
    
    def Vg_PESCA_20(self, rho, epsilon = 12, ns = 1e15, d1 = 0, d2 = 0, d4 = 0, nd = 1e15):
        dtot = d1 + d2 + d4
        res =  -(((dtot)/epsilon)*self.epsilon_0)*(-1*self.ee*ns) + (((d2/2)+(d4/2))/(epsilon*self.epsilon_0))*-1*self.ee*nd
        return res

    def Vg_20(self, rho, epsilon = 12, ns = 1e15, d1 = 0, d2 = 0, d4 = 0, nd = 1e15):
        dtot = d1 + d2 + d4
        res =  (-((1/(-1*self.ee**2*rho)) + ((dtot)/epsilon))*(-1*self.ee*ns)) + (((d2/2)+(d4/2)/(epsilon*self.epsilon_0))*-1*self.ee*nd)
        return res