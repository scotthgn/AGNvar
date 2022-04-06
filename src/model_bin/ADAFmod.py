#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  1 15:26:28 2022

@author: wljw75
"""

"""
Models for the advenction dominated accretion flow - or in other words
the hot inner corona.

Currently only one model - which is a hot Comptonised corona
See Kubota & Done (2018)
"""

import numpy as np
import astropy.units as u
import astropy.constants as constants
from .pyNTHCOMP import donthcomp
from .DiscMods import Disc, CompDisc


"""Usefull constants"""
h = constants.h.value #Js - Planck constant
G = constants.G.value * constants.M_sun.value #m^3/(M_sol^1 s^2) - Gravitational constant
sigma = constants.sigma_sb.value #W/m^2K^4 - Stefan-Boltzmann constant
k_B = constants.k_B.value #J/K - Boltzmann constant
c = constants.c.value #m/s - Speed of light
mp = constants.m_p.value #kg - Proton mass
sigmaT = constants.sigma_T.value #m^-2 - Thomson cross section
Ce = constants.e.value #C - elementary charge



class ADAF(Disc):
    """
    An advection dominated accretion flow - i.e Hot corona
    Currently only calculates this as in Kubota & Done (2018)
    """
    
    def __init__(self,
                 r_hot,
                 M,
                 inc,
                 mdot,
                 a_star,
                 gamma_h,
                 kT_h,
                 r_out,
                 seed_type='WC',
                 r_w=100,
                 kT_w=0.2,
                 gamma_w=2.1):
        """
        Initiates ADAS

        Parameters
        ----------
        r_hot : float
            Outer radius of hot fluffy Corona - units : Rg.
        M : float
            Black Hole mass - units : Msol.
        inc : float
            Inclination angle - units : deg.
        mdot : float
            Mass accretion rate - Eddington.
        a_star : float
            BH spin - dimensionless.
        gamma_h : float
            Photon index for hot corona.
        kT_h : float
            Electron temperature of hot corona (high energy roll over)
            units : keV.
        r_out : float
            Outer radius of accretion disc - units : Rg.
        seed_type : str, optional
            WC or AD - says whether the disc is thermalised or not at trunction
            radius. The default is 'WC'.
        r_w : float, optional
            IF seed_type WD, need to define warm compton radius. 
            The default is 100. Units : Rg
        kT_w : float, optional
            IF seed_type WD, need to define electron temperature of warm Compton.
            The default is 0.2. Units : keV
        gamma_w : float, optional
            IF seed_type WD, need to define photon index of warm Compton.
            The default is 2.1.

        """
        
        #read params
        self.r_h = r_hot
        self.M = M
        self.inc = np.deg2rad(inc)
        self.mdot = mdot
        self.a = a_star
        self.gamma_h = gamma_h
        self.kT_h = kT_h
        self.r_out = r_out
        self.seed_type = seed_type
        self.r_w = r_w
        self.kT_w = kT_w
        self.gamma_w = gamma_w
        
        
        #Initiating Disc - as need seed photon luminosity
        Disc.__init__(self, r_hot, r_out, a_star, inc, mdot, M, model='AD')
        
        #creating warm compton model IF seed type is warm compton...
        if self.seed_type == 'WC':
            self.wc_mod = CompDisc(r_hot, r_out, a_star, inc, mdot, M, gamma_w, kT_w)
            
        
        self._T_seed()
        self._L_seed()
        
        
    """
    Calculating seed photon parameters - i.e luminosity and temperature
    """
    def _T_seed(self):
        """
        Calculates seed photon temperature. Assumes typical T is same as 
        from inner edge of the disc, so T_NT(r_hot) for standard accretion
        disc or T_NT(r_hot) * exp(y_w), where y_w is Compron y-parameter
        for the warm comptonisation region (see Kubota & Done 2018)
        """
        if self.seed_type == 'AD':
            #Ehhhm, NO. Fix this - Remeber to mask below r_tr!!
            T_int = self.Td[0, 0] #Inner disc temperature in K
            T_irr = self.T_rep(self.Lx)
            T_irr = T_irr[0, 0]
            
            Ts = (T_int**4 + T_irr**4)**(1/4)
            
            kTs = k_B * Ts
            self.kT_seed = (kTs * u.J).to(u.keV).value
        
        else:
            #Using the soft-compton spectrum to estimate temperature,
            #seeing as don't know optical depth \tau.
            #Assuming equilibrium temperature given by inverse Compton temp.
            wc_spec = self.wc_mod.Calc_spec(self.Lx)
            wc_L = np.trapz(wc_spec, self.nu_grid)
            wc_E = np.trapz(h*self.nu_grid*wc_spec, self.nu_grid)
            
            kT_ic = wc_E/(4 * wc_L)
            self.kT_seed = (kT_ic * u.J).to(u.keV).value
    
    
    def _L_seed(self):
        T_int = self.Td
        T_rep = self.T_rep(self.Lx)
        
        T_int = np.ma.masked_where(self.d_mask, T_int)
        T_rep = np.ma.masked_where(self.d_mask, T_rep)
        
        #Flattening since no phi dependence
        Tint_flat = T_int[0, :]
        Trep_flat = T_rep[0, :]        
        R_flat = self.R_grid[0, :]
        
        T_tot4 = (Tint_flat**4 + Trep_flat**4)
        Ftot_r = sigma * T_tot4
        
        theta_0 = np.arcsin(self.Hx/R_flat)
        cov_fac = theta_0 - 0.5 * np.sin(2*theta_0)
        
        self.Lseed = 2 * np.trapz(Ftot_r * (cov_fac/np.pi) * 2*np.pi * R_flat, R_flat)
      
        
    
    
    """
    Calculating spectrum
    """
    def Calc_spec(self):
        """
        Calculates spectrum in units W/Hz

        """
        self.Es = (self.nu_grid * u.Hz).to(u.keV, equivalencies=u.spectral()).value
        ph_hc = donthcomp(self.Es, [self.gamma_h, self.kT_h, self.kT_seed, 1, 0])
        ph_hc = (ph_hc * u.W/u.Hz).to(u.W/u.keV, equivalencies=u.spectral()).value  
        
        #normC = (self.Lseed + self.Lx)/np.trapz(ph_hc, self.nu_grid)
        normC = self.Lx/np.trapz(ph_hc, self.nu_grid)
        L_nu = normC * ph_hc #* np.cos(self.inc)
                
        
        return L_nu

            
        
        
        
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    
    r_h = 25
    r_w = 100
    r_out = 400
    M = 2e8
    inc = 25
    mdot = 10**(-1.4)
    a = 0
    gamma_h = 1.8
    kT_h = 100
    seed_type='WC'
    kT_w = 0.2
    gamma_w = 2.1
    
    
    adaf_mod = ADAF(r_h, M, inc, mdot, a, gamma_h, kT_h, r_out)
    Lnus = adaf_mod.Calc_spec()
    nus = adaf_mod.nu_grid
    
    plt.loglog(nus, nus*Lnus)
    plt.show()
            