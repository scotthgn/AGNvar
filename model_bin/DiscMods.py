#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  2 10:12:25 2022

@author: wljw75
"""

"""
Creates one of three possible accretion disc objects: Standard disc according
to Shakura & Sunnyaev (1973), a disc with some darkening radius, and a 
Comptonised disc.
Includes methods for creating both spectra, and evolving the light-curve of 
a single model component
"""

import numpy as np
import astropy.units as u
from functools import partial
from multiprocessing import Pool
from tqdm import tqdm
import warnings

#Stop all the run-time warnings (we know why they happen - doesn't affect the output!)
warnings.filterwarnings('ignore') 


"""Usefull constants"""
h = 6.63e-34 #Js - Planck constant
G = 6.67e-11 * 1.99e+30 #m^3/(M_sol^1 s^2) - Gravitational constant
sigma = 5.67e-8 #W/m^2K^4 - Stefan-Boltzmann constant
k_B = 1.38e-23 #J/K - Boltzmann constant
c = 3e+8 #m/s - Speed of light
mp = 1.67e-27 #kg - Proton mass
sigmaT = 6.653e-29 #m^-2 - Thomson cross section
Ce = 1.6e-19 #C - elementary charge




class Disc:
    """
    Creates the spectra for a disc with some darkening radius r_dark
    For a given nu/E will also evolve spectra for an input light-curve
    """
    
    def __init__(self, r_d, r_out, r_isco, hx, inc, mdot, eta, M, A, Emin, Emax):
        """
        Initiates spec object

        Parameters
        ----------
        r_d : float
            Darkening radius (Similar to truncation) - units : Rg.
        r_out : float
            Outer disc radius - units : Rg.
        r_isco : float
            Inner most stable circular orbit (depends on a) - units : Rg.
        hx : float
            Height of X-ray corona - units : Rg.
        inc : float
            System inclination - units : deg.
        mdot : float
            Eddington accretion rate.
        eta : float
            Accretion efficiency (from L = eta * Mdot * c^2)
        M : float
            BH mass.
        A : float
            Disc albedo - must be between 0 and 1
        Emin : float
            Min energy to use when calculating power - units : keV
        Emax : float
            Max energy to use when calculating power - units : keV
        
        """
        #Read params
        self.r_d = r_d
        self.r_out = r_out
        self.r_isco = r_isco
        self.hx = hx
        self.inc = np.deg2rad(inc)
        self.mdot = mdot
        self.eta = eta
        self.M = M
        self.A = A
        self.Emin = Emin
        self.Emax = Emax
        
        
        #Conversion factors to physical units
        self.Rg = (G*self.M)/c**2 #Grav radius for this system in meters
        self.Mdot_edd = (4*np.pi*G*self.M*mp)/(self.eta*sigmaT*c)
        
        
        #Physical units
        self.R_d = self.r_d * self.Rg
        self.R_out = self.r_out * self.Rg
        self.R_isco = self.r_isco * self.Rg
        self.Hx = self.hx * self.Rg
        self.Mdot = self.mdot * self.Mdot_edd
        
        
        #Creating grid over disc
        dlog_R = (np.log10(self.R_out) - np.log10(self.R_isco))/400
        R_inMidBin = 10**(np.log10(self.R_isco) + dlog_R/2)
        R_outMidBin = 10**(np.log10(self.R_out) - dlog_R/2)
        
        self.R_grid, self.phi_grid = np.meshgrid(np.geomspace(
            R_inMidBin, R_outMidBin, 400), np.linspace(0, 2*np.pi, 400))
        
        
        #Setting up energy grid for power calculations
        nu_min = (self.Emin * u.keV).to(u.Hz, equivalencies=u.spectral())
        nu_max = (self.Emax * u.keV).to(u.Hz, equivalencies=u.spectral())
        
        self.nu_min = nu_min.value
        self.nu_max = nu_max.value
        self.nu_grid = np.geomspace(self.nu_min, self.nu_max, 1000)
        
    

    #The delay surface on the disc (following Welsh & Thorne 1991)
    def delay_surf(self):
        #Delay surface in seconds
        tau_sec = (1/c) * (np.sqrt(self.R_grid**2 + self.Hx**2) + 
                           self.Hx*np.cos(self.inc) - 
                           self.R_grid*np.cos(self.phi_grid)*np.sin(self.inc)) 
    
        tau = tau_sec/(60*60*24) # Delay surface in days
        return tau



    """
    Section for the Temperatures - one function for T_intrinsic (T_int)
    one for T_reprocessed (T_rep)
    """
    
    def T_int(self):
       
        T4 = ((3*G*self.M*self.Mdot)/(8*np.pi*sigma*self.R_grid**3)) * (
            1 - np.sqrt(self.R_isco/self.R_grid))  
        
        return T4**(1/4)
    

    def T_rep(self, Lxs):
        """
        Re-processed temperaure

        Parameters
        ----------
        Lxs : float OR array (same shape as R_grid)
            X-ray luminosity seen by point on disc.

        Returns
        -------
        T_rep : float OR array
            Reprocessed disc temperature - units : K
        """
        
        T4_rep = (Lxs * self.Hx)/(4*np.pi * sigma * 
                                  (self.R_grid**2 + self.Hx**2)**(3/2)) 
        
        return ((1 - self.A) * T4_rep)**(1/4)
    
    
    
    """
    Section for intrinsic luminosities - used to calculate mean X-ray 
    power
    """
    
    def int_power(self, r_cut):
        """
        Total luminosity for a Shakura-Sunyaev accretion disc

        Parameters
        ----------
        r_cut : float
            Cut off radius to calculate to - units : Rg.

        Returns
        -------
        Ltot : float
            Bolometric luminosity of disc - units : W.

        """      
        R_cut = r_cut * self.Rg
        Td_all = self.T_int()
        
        #Masking grid values below r_cut
        R_mask = np.ma.masked_less_equal(self.R_grid, R_cut)
        Td = np.ma.masked_where(np.ma.getmask(R_mask), Td_all)
        
        Td_r = Td[0, :] #Flattening since no phi dependence on T
        R_flat = R_mask[0, :]
        
        
        #Planck law for black body
        B_nu = ((2*h*self.nu_grid**3)/c**2) * (
            1/(np.exp((h*self.nu_grid)/(k_B * Td_r[:, np.newaxis])) - 1))
        
        #Converting from intensity to photon flux
        Fnu = np.pi * B_nu
        dLnu = 2 * 2*np.pi * Fnu * R_flat[:, np.newaxis]
        Lnu = np.trapz(y=dLnu * np.cos(self.inc), x=R_flat, axis=0)
        
        Ltot = np.trapz(y=Lnu, x=self.nu_grid)
        
        return Ltot

    
    
    def calc_XrayPower(self):
        L_full = self.int_power(self.r_isco)
        L_tr = self.int_power(self.r_d)
        
        Lxr = L_full - L_tr
        return Lxr
    
    
    
    """
    Function for calculating luminosity in given band. Dark and truncated
    versions
    """
    
    def calc_LnuDark(self, nu, Lirr):
        
        T_intAll = self.T_int()
        T_repAll = self.T_rep(Lirr)
        
        #Applying mask to T_int for dark section of disc
        R_mask = np.ma.masked_less_equal(self.R_grid, self.R_d)
        T_intMa = np.ma.masked_where(np.ma.getmask(R_mask), T_intAll)
        T_intMa = T_intMa.filled(0) #Placing 0 on mask so adds properly to Trep
        
        T_tot = (T_intMa**4 + T_repAll**4)**(1/4)
        
        B_nu = ((2*h*nu**3)/c**2) * (
            1/(np.exp((h*nu)/(k_B * T_tot)) -1))
        
        Fnu = np.pi * B_nu
        dLs_r = 2 * np.trapz(y=Fnu, x=self.phi_grid[:, 0], axis=0)
        Lnu = np.trapz(y=dLs_r * np.cos(self.inc) * self.R_grid[0, :], 
                        x=self.R_grid[0, :])
        
        return Lnu
    
    
    def calc_LnuTrunc(self, nu, Lirr):
        T_intAll = self.T_int()
        T_repAll = self.T_rep(Lirr)
        
        #Applying mask to T_int for dark section of disc
        R_mask = np.ma.masked_less_equal(self.R_grid, self.R_d)
        T_intMa = np.ma.masked_where(np.ma.getmask(R_mask), T_intAll)
        T_repMa = np.ma.masked_where(np.ma.getmask(R_mask), T_repAll)
        
        T_tot = (T_intMa**4 + T_repMa**4)**(1/4)
        
        B_nu = ((2*h*nu**3)/c**2) * (
            1/(np.exp((h*nu)/(k_B * T_tot)) -1))
        
        Fnu = np.pi * B_nu
        dLs_r = 2 * np.trapz(y=Fnu, x=self.phi_grid[:, 0], axis=0)
        Lnu = np.trapz(y=dLs_r * np.cos(self.inc) * self.R_grid[0, :], 
                        x=self.R_grid[0, :])
        
        return Lnu
    
    
    
    """
    Functions for evolving light-curves
    Will have both for single value of nu version, and a multiple value 
    version using multiprocessing for speedy analysis
    """
    
    def do_evolve(self, nu, l_xs, ts, mode='dark'):
        """
        Evolves the system according to input X-ray light-curve

        Parameters
        ----------
        nu : float
            Frequency to calculate model light-curve at - units : Hz.
        l_xs : 1D-array
            Fractional X-rays - units : Lx/mean(Lx).
        ts : 1D-array
            Corresponding time coords - units : days.
        mode : str
            dark or trunc - Tells code whether to model dark disc or truncated
            disc

        Returns
        -------
        Lcurve_mod : 1D-array
            Model light-curve after evolving system.

        """
        
        #Ensuring t starts at 0 - else it mucks up calculation...
        try:
            assert ts[0] == 0
        except:
            ts = ts - ts[0]
        
        #Checking valid mode (dark or trunc)
        try:
            assert mode == 'dark' or mode == 'trunc'
        except:
            print('Not a valid mode!')
            print('Valid modes are: dark OR trunc')
            exit()
        
        tau_surf = self.delay_surf()
        Lx_mean = self.calc_XrayPower()
        
        Lxs = l_xs * Lx_mean #array of x-ray lums
        Lin = np.array([Lx_mean]) #array of Ls in play
        
        Lirr = np.ndarray(np.shape(tau_surf)) #irradiation array
        Lcurve_mod = np.array([]) #Predicted light curve output
        for i in tqdm(range(len(ts))): #tqdm gives progress bar
            
            Lin = np.append(Lin, [Lxs[i]])
            if i == 0:
                t_delay = np.array([np.inf])
            else:
                t_delay += ts[i] - ts[i-1] #Adding time step to delay array
            
            t_delay = np.append(t_delay, [0]) #Appending 0 for current emitted
            
            #Sorting irradiation array
            for j in range(len(t_delay)):
                Lirr[tau_surf <= t_delay[j]] = Lin[j]
            
            
            #Calculating Lnu based off model input string
            if mode == 'dark':
                Lnu_t = self.calc_LnuDark(nu, Lirr)
            elif mode == 'trunc':
                Lnu_t = self.calc_LnuTrunc(nu, Lirr)
                
            Lcurve_mod = np.append(Lcurve_mod, [Lnu_t])
        
        return Lcurve_mod
            
    
    
    def multi_evolve(self, nus, l_xs, ts, numCPU, mode='dark'):
        """
        Multithreads do_evolve() - allows for calculation of multiple 
        frequencies simoultaneously

        Parameters
        ----------
        nus : 1D-array
            Input array of frequencies to calculate l-curve for - units : Hz.
        l_xs : 1D-array
            See do_evolve().
        ts : 1D-array
            See do_evolve().
        numCPU : float
            Number of CPU cores to use.

        Returns
        -------
        all_Ls : 2D_array
            All output light-curves stacked in columns.

        """
        
        evolve_func = partial(self.do_evolve, l_xs=l_xs, ts=ts, mode=mode)
        with Pool(numCPU) as p:
            l_curves = p.map(evolve_func, nus)
            
            for k in range(len(nus)):
                if k == 0:
                    all_Ls = l_curves[k]
                else:
                    all_Ls = np.column_stack((all_Ls, l_curves[k]))
        
        return all_Ls
    
    
    
    """
    Section for calculating the spectra. Seperate method for each model
    (currently just truncated disc (AD_spec) and dark disc (darkAD_spec))
    """
    
    def AD_spec(self, r_tr):
        """
        Calculates spec for standard accretion disc - usefull for comparing
        to dark disc model

        Parameters
        ----------
        r_tr : float
            Truncation radius - units : Rg.

        Returns
        -------
        Lnu : 1D-array
            Output luminosities for spectra - units : W.

        """
        
        Lx = self.calc_XrayPower() #For irradiating the disc
        
        Rcut = r_tr * self.Rg
        Tint_all = self.T_int()
        Trep_all = self.T_rep(Lx)
        
        #Masking grid values below r_tr
        R_mask = np.ma.masked_less_equal(self.R_grid, Rcut)
        Td = np.ma.masked_where(np.ma.getmask(R_mask), Tint_all)
        Tr = np.ma.masked_where(np.ma.getmask(R_mask), Trep_all)
        
        #Flattening since no phi dependence
        Tint_flat = Td[0, :]
        Trep_flat = Tr[0, :]        
        R_flat = R_mask[0, :]
      
        T_tot = (Tint_flat**4 + Trep_flat**4)**(1/4)
        
        B_nu = ((2*h*self.nu_grid**3)/c**2) * (
            1/(np.exp((h*self.nu_grid)/(k_B * T_tot[:, np.newaxis])) - 1))
        
        F_nu = np.pi * B_nu
        dLnu = 2 * 2*np.pi * F_nu * R_flat[:, np.newaxis]
        Lnu = np.trapz(y=dLnu * np.cos(self.inc), x=R_flat, axis=0)
        
        return Lnu
    
    
    def darkAD_spec(self, rd):
        """
        Calculates spectra for accretion disc, with dark inner region

        Parameters
        ----------
        rd : float
            Darkening radius - units : Rg.

        Returns
        -------
        Lnu : 1D-array
            Output luminosities for spectra - units : W.

        """
        
        Lx = self.calc_XrayPower()
        
        Rcut = rd * self.Rg
        Tint_all = self.T_int()
        Trep_all = self.T_rep(Lx)
        
        #Masking grid values for intrinsic below rd
        R_mask = np.ma.masked_less_equal(self.R_grid, Rcut)
        Td = np.ma.masked_where(np.ma.getmask(R_mask), Tint_all)
        Td = Td.filled(0)
        
        #Flattening since no phi dependence
        Tint_flat = Td[0, :]
        Trep_flat = Trep_all[0, :]
        R_flat = self.R_grid[0, :]
        
        T_tot = (Tint_flat**4 + Trep_flat**4)**(1/4)
        
        B_nu = ((2*h*self.nu_grid**3)/c**2) * (
            1/(np.exp((h*self.nu_grid)/(k_B * T_tot[:, np.newaxis])) - 1))
        
        F_nu = np.pi * B_nu
        dLnu = 2 * 2*np.pi * F_nu * R_flat[:, np.newaxis]
        Lnu = np.trapz(y=dLnu * np.cos(self.inc), x=R_flat, axis=0)
        
        return Lnu
        
        
        
    

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    
    r_d = 25
    r_out = 1e4
    r_isco = 6
    hx = 10
    inc = 20
    mdot = 10**(-1.4)
    eta = 0.1
    M = 2e8
    
    sp = Disc(r_d, r_out, r_isco, hx, inc, mdot, eta, M, 0.5, 1e-4, 0.1)
    
    ts_test = np.array([0, 1, 2, 3, 4])
    xlfrac = np.array([1, 1.2, 1.3, 1.1, 0.8])
    nus = np.array([1e14, 1e15, 1e13])
    
    print(sp.calc_XrayPower())
    
    #lm = sp.multi_evolve(nus, xlfrac, ts_test, 3)
    #print(lm)
    
   
    #examining the spectra
    s_fullDisc = sp.AD_spec(6)
    s_truncDisc = sp.AD_spec(r_d)
    s_darkDisc = sp.darkAD_spec(r_d)
    nus = sp.nu_grid
    
    #Converting to erg
    s_fd = (s_fullDisc * u.W).to(u.erg/u.s).value
    s_td = (s_truncDisc * u.W).to(u.erg/u.s).value
    s_dd = (s_darkDisc * u.W).to(u.erg/u.s).value
    
    print(max(s_fd), max(s_td))
    plt.loglog(nus, nus * s_fd, label='AD r_isco')
    plt.loglog(nus, nus * s_td, label='AD r_tr=25')
    plt.loglog(nus, nus * s_dd, label='dark AD r_d=25')
    
    plt.ylim(1e43, 1e45)
    plt.xlim(6e13, 2e16)
    
    plt.ylabel(r'$\nu F_{\nu}$   erg/s')
    plt.xlabel(r'$\nu$   Hz')
    
    #Indicating bands used for obs
    bands = np.array(['UVW2', 'UVM2', 'UVW1', 'U', 'B', 'V'])
    wvl = np.array([1928, 2246, 2600, 3465, 4392, 5468])
    nus_b = (wvl * u.AA).to(u.Hz, equivalencies=u.spectral()).value

    for n in range(len(bands)):
        plt.axvline(nus_b[n], ls='dotted')
    
    
    plt.legend()
    plt.show()
        
    