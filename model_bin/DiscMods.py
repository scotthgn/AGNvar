#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  2 10:12:25 2022

@author: wljw75
"""

"""
Creates one of three possible accretion disc objects: Standard disc according
to Shakura & Sunnyaev (1973) with a Novikov-Thorne temperature profile
(Novikov & Thorne 1973), a disc with some darkening radius, and a 
Comptonised disc.
Includes methods for creating both spectra, and evolving the light-curve of 
a single model component
"""

import numpy as np
import astropy.units as u
import astropy.constants as constants
from functools import partial
from multiprocessing import Pool
from tqdm import tqdm
import warnings
import excTHCOMP as thc
from pyNTHCOMP import donthcomp

#Stop all the run-time warnings (we know why they happen - doesn't affect the output!)
warnings.filterwarnings('ignore') 


"""Usefull constants"""
h = constants.h.value #Js - Planck constant
G = constants.G.value * constants.M_sun.value #m^3/(M_sol^1 s^2) - Gravitational constant
sigma = constants.sigma_sb.value #W/m^2K^4 - Stefan-Boltzmann constant
k_B = constants.k_B.value #J/K - Boltzmann constant
c = constants.c.value #m/s - Speed of light
mp = constants.m_p.value #kg - Proton mass
sigmaT = constants.sigma_T.value #m^-2 - Thomson cross section
Ce = constants.e.value #C - elementary charge





class Disc:
    """
    Creates a disc object. Used to create spectra, or to evolve the disc
    according to a light-curve according to standard accretion disc.
    
    To possible models: truncated (AD) and darkened (DD). 
    
    AD corresponds to a standard Shakura-Sunyaev disc, truncated at some 
    radius. If you want the disc to extend to r_isco, simply set r_in = r_isco
    
    DD corresponds to a standard AD down to a darkening radius (r_in), beyon
    which all the intrinsic power is transported away through some magical
    process (probabliy magnetic fields or something). Hence the only 
    contribution to the flux from within the darkening radius will be the 
    reflected component.
    
    """
    Emin = 1e-4 #keV
    Emax = 10 #keV
    hx = 10 #height of corona
    A = 0.3 #disc albedo - fixed same as in AGNSED
    numR = 400 #Nr of gridpoints in R
    numphi = 400 #Nr of gridpoints in phi
    
    def __init__(self, r_in, r_out, a_star, inc, mdot, M, model='AD'):
        """
        Initiates spec object

        Parameters
        ----------
        r_in : float
            Inner radius of disc - units : Rg.
        r_out : float
            Outer disc radius - units : Rg.
        a_star : float
            Dimensionless spin parameter. +ve for prograde rotation, 
            -ve for retrograde - range [-1, 1].
        inc : float
            System inclination - units : deg.
        mdot : float
            Eddington accretion rate.
        M : float
            BH mass - units : Msol.
        model : str
            Model to use, possibilities are: AD (standard accretion disc), 
            DD (Darkened accretion disc)
        
        """
        #Read params
        self.r_in = r_in
        self.r_out = r_out
        self.a = a_star
        self.inc = np.deg2rad(inc)
        self.mdot = mdot
        self.M = M
        self.mod = model
        
        
        #Calculating disc params
        self._calc_Ledd() #eddington luminosity
        self._calc_risco() #innermost stable curcular orbit
        self._calc_efficiency() #accretion efficiency
        
        #Performing checks
        self._check_mod()
        self._check_inc()
        self._check_risco()
        self._check_rlims()
        

        #Conversion factors to physical units
        self.Rg = (G*self.M)/c**2 #Grav radius for this system in meters
        self.Mdot_edd = self.L_edd/(self.eta * c**2)
        
        
        #Physical units
        self.R_in = self.r_in * self.Rg
        self.R_out = self.r_out * self.Rg
        self.R_isco = self.r_isco * self.Rg
        self.Hx = self.hx * self.Rg
        self.Mdot = self.mdot * self.Mdot_edd
        
        
        #Creating grid over disc
        dlog_R = (np.log10(self.R_out) - np.log10(self.R_isco))/self.numR
        R_inMidBin = 10**(np.log10(self.R_isco) + dlog_R/2)
        R_outMidBin = 10**(np.log10(self.R_out) - dlog_R/2)
        
        self.R_grid, self.phi_grid = np.meshgrid(np.geomspace(
            R_inMidBin, R_outMidBin, self.numR), np.linspace(0, 2*np.pi, self.numphi))
        
        #Creating relevant mask over grid - to accounts for truncation radii
        self.d_mask = np.ma.getmask(np.ma.masked_less_equal(self.R_grid, self.R_in))
        #print(self.d_mask)
        
        
        #Setting up energy grid for power calculations
        nu_min = (self.Emin * u.keV).to(u.Hz, equivalencies=u.spectral())
        nu_max = (self.Emax * u.keV).to(u.Hz, equivalencies=u.spectral())
        
        self.nu_min = nu_min.value
        self.nu_max = nu_max.value
        self.nu_grid = np.geomspace(self.nu_min, self.nu_max, 1000)
        
        
        #Disc properties that don't change during evolution
        self.tau_grid = self.delay_surf() #delay surface
        self.Td = self.T_int() #intrinisc disc temperature
        self.Lx = self.calc_XrayPower() #Mean X-ray power

    
    
    """
    Performing checks on certain input variables
    (i.e Making sure they conform, so don't brake code!!!)
    """
    
    def _check_mod(self):
        if self.mod == 'AD' or self.mod == 'DD':
            pass
        else:
            raise AssertionError('Wrong model input \n'
                  'Must be AD (accretion disc), or \n'
                  'DD (darkened accretion disc) \n')
    
    def _check_inc(self):
        if self.inc >= 0 and self.inc <= np.pi/2:
            pass
        else:
            raise AssertionError('Inclination outside hard range [0, 90] deg \n')
    
    def _check_risco(self):
        if self.r_in >= self.r_isco:
            pass
        else:
            raise AssertionError('r_in < r_isco - Not physically permitted!!')
    
    def _check_rlims(self):
        if self.r_out >= 2 * self.r_in:
            pass
        else:
            raise AssertionError('r_out < 2*r_in -- WARNING!!! \n'
                                 'Insufficient spacing between r_out and r_in \n'
                                 'on grid! - Increase r_out or reduce r_in to \n'
                                 'satisfy this criteria')
    
        
            
    
    
    
    """
    Section for calculating disc parameters. i.e r_isco, effeiciency eta,
    NT-parameters, delay surface, etc
    
    """
    def _calc_Ledd(self):
        """
        Caclulate eddington Luminosity
        """
        Ledd = (4 * np.pi * G * self.M * mp * c)/sigmaT
        self.L_edd = Ledd
    
    
    def _calc_risco(self):
        """
        Calculating innermost stable circular orbit for a spinning
        black hole. Follows Page and Thorne (1974). Note, can also be reffered
        to as r_ms, for marginally stable orbit
        
        return r_isco as property - so will be called in __init__

        """
        Z1 = 1 + (1 - self.a**2)**(1/3) * (
            (1 + self.a)**(1/3) + (1 - self.a)**(1/3))
        Z2 = np.sqrt(3 * self.a**2 + Z1**2)

        self.r_isco = 3 + Z2 - np.sign(self.a) * np.sqrt(
            (3 - Z1) * (3 + Z1 + 2*Z2))

    
    
    def _calc_efficiency(self):
        """
        Calculates the accretion efficiency eta, s.t L_bol = eta Mdot c^2
        Using the GR case, where eta = 1 - sqrt(1 - 2/(3 r_isco)) 
            Taken from: The Physcis and Evolution of Active Galactic Nuceli,
            H. Netzer, 2013, p.38
        
        Note to self!: When I derive this in Newtonian limit I get
        eta = 1/(2 r_isco). Not entirely sure how to derive the GR version.
        Should ask Chris at next meeting!!!!

        """
        
        self.eta = 1 - np.sqrt(1 - 2/(3*self.r_isco))
    
    
    def _calc_NTparams(self, r):
        """
        Calculates the Novikov-Thorne relativistic factors.
        see Active Galactic Nuclei, J. H. Krolik, p.151-154
        and Page & Thorne (1974)

        """
        y = np.sqrt(r)
        y_isc = np.sqrt(self.r_isco)
        y1 = 2 * np.cos((1/3) * np.arccos(self.a) - (np.pi/3))
        y2 = 2 * np.cos((1/3) * np.arccos(self.a) + (np.pi/3))
        y3 = -2 * np.cos((1/3) * np.arccos(self.a))

        
        B = 1 - (3/r) + ((2 * self.a)/(r**(3/2)))
        
        C1 = 1 - (y_isc/y) - ((3 * self.a)/(2 * y)) * np.log(y/y_isc)
        
        C2 = ((3 * (y1 - self.a)**2)/(y*y1 * (y1 - y2) * (y1 - y3))) * np.log(
            (y - y1)/(y_isc - y1))
        C2 += ((3 * (y2 - self.a)**2)/(y*y2 * (y2 - y1) * (y2 - y3))) * np.log(
            (y - y2)/(y_isc - y1))
        C2 += ((3 * (y3 - self.a)**2)/(y*y3 * (y3 - y1) * (y3 - y2))) * np.log(
            (y - y3)/(y_isc - y3))
        
        C = C1 - C2
        
        return C/B
    
    
    
    """
    Now onto the acutal model
    """
    
    #Calculate time delay across disc
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
       
        Rt = self._calc_NTparams(self.R_grid/self.Rg)
        const_fac = (3 * G * self.M * self.mdot * self.Mdot_edd)/(
            8 * np.pi * sigma * (self.R_grid)**3)
        
        T4 = const_fac * Rt 
        
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
        #Using different masking than the one initiated in __init__
        #Since this function will be used for more than just r_in
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
        L_tr = self.int_power(self.r_in)
        
        Lxr = L_full - L_tr
        return Lxr
    
    
    
    
    
    
    """
    Function for calculating luminosity in given band. Dark and truncated
    versions
    """
    
    def calc_Lnu(self, nu, Lirr):
        
        T_int = self.Td
        T_rep = self.T_rep(Lirr)
        
        #Applying mask to T grids (model dependent) to account for darkening
        #or truncations
        T_int = np.ma.masked_where(self.d_mask, T_int)
        
        if self.mod == 'AD':
            T_rep = np.ma.masked_where(self.d_mask, T_rep)
        else:
            T_int = T_int.filled(0) #Placing 0 on mask so adds properly to Trep
        
        T_tot = (T_int**4 + T_rep**4)**(1/4)
        
        
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
    
    def do_evolve(self, nu, l_xs, ts):
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
        

        
        Lxs = l_xs * self.Lx #array of x-ray lums
        Lin = np.array([self.Lx]) #array of Ls in play
        
        Lirr = np.ndarray(np.shape(self.tau_grid)) #irradiation array
        Lcurve_mod = np.array([]) #Predicted light curve output
        #for i in tqdm(range(len(ts))): #tqdm gives progress bar
        for i in range(len(ts)):
            
            Lin = np.append(Lin, [Lxs[i]])
            if i == 0:
                t_delay = np.array([np.inf])
            else:
                t_delay += ts[i] - ts[i-1] #Adding time step to delay array
            
            t_delay = np.append(t_delay, [0]) #Appending 0 for current emitted
            
            #Sorting irradiation array
            for j in range(len(t_delay)):
                Lirr[self.tau_grid <= t_delay[j]] = Lin[j]
            
            #Calculating Lnu for current time step
            Lnu_t = self.calc_Lnu(nu, Lirr)            
            Lcurve_mod = np.append(Lcurve_mod, [Lnu_t])
        
        return Lcurve_mod
            
    
    
    def multi_evolve(self, nus, l_xs, ts, numCPU):
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
        
        evolve_func = partial(self.do_evolve, l_xs=l_xs, ts=ts)
        with Pool(numCPU) as p:
            l_curves = p.map(evolve_func, nus)
            
            for k in range(len(nus)):
                if k == 0:
                    all_Ls = l_curves[k]
                else:
                    all_Ls = np.column_stack((all_Ls, l_curves[k]))
        
        return all_Ls
    
    
    
    def evolve_fullSpec(self, l_xs, ts, num_nuBin, numCPU):
        """
        Evolve the entire spectrum according to input X-ray light-curve
        NOTE! Computationally intense!!!
        Parameters
        ----------
        l_xs : 1D-array
            See do_evolve().
        ts : 1D-array
            See do_evolve().
        numCPU : float
            Number of CPU cores to use.
        num_nuBin : float
            Number of frequency bins to use between nu_min and nu_max. 
            Set to -1 to use self.nu_grid (note that this will be
            computationally intense, and impractical for most applications)

        Returns
        -------
        all_Ls : 2D_array
            All output light-curves stacked in columns.
        nus : 1D-array
            Corresponding frequencies for spectra

        """
        
        if num_nuBin == -1:
            nus = self.nu_grid
        else:
            nus = np.geomspace(self.nu_min, self.nu_max, num_nuBin)
        
        all_Ls = self.multi_evolve(nus, l_xs, ts, numCPU)
        
        return all_Ls, nus
    
    
    
    
    """
    Section for calculating the spectra.
    """
    
    def Calc_spec(self):
        """
        Calculates spec for current accretion disc model 

        Returns
        -------
        Lnu : 1D-array
            Output luminosities for spectra - units : W.

        """
        
        Lx = self.Lx #For irradiating the disc

        Tint = self.Td
        Trep = self.T_rep(Lx)
        
        #Masking grid values below r_tr/r_d
        Tint = np.ma.masked_where(self.d_mask, Tint)
        if self.mod == 'AD':
            Trep = np.ma.masked_where(self.d_mask, Trep)
        else:
            Tint = Tint.filled(0) #Placing 0 on mask so adds properly to Trep
        
        
        #Flattening since no phi dependence
        Tint_flat = Tint[0, :]
        Trep_flat = Trep[0, :]        
        R_flat = self.R_grid[0, :]
      
        T_tot = (Tint_flat**4 + Trep_flat**4)**(1/4)
        
        B_nu = ((2*h*self.nu_grid**3)/c**2) * (
            1/(np.exp((h*self.nu_grid)/(k_B * T_tot[:, np.newaxis])) - 1))
        
        F_nu = np.pi * B_nu
        #F_nu = B_nu
        dLnu = 2 * 2*np.pi * F_nu * R_flat[:, np.newaxis]
        Lnu = np.trapz(y=dLnu * np.cos(self.inc), x=R_flat, axis=0)
        
        
        return Lnu
    
    



class CompDisc(Disc):
    """
    A comptonised accretion disc. Calculates the seed photons as a black-body
    from a disc, then uses pyNTHCOMP; python version of the XSPEC model NTHCOMP
    (Zdziarski, Johnson & Magdziarz, 1996; Zycki, Done & Smith, 1999). Adapted 
    by Thomas et al. 2016.
    
    Note! A previous version of the code calculated the entire disc spectrum, 
    and then convolved it with THCOMP to calculate the Comptonised spectrum. 
    However, this is uneccesarily slow - hence the change!
    """
    
    def __init__(self, r_in, r_out, a_star, inc, mdot, M, gamma_c, kTe_c):
        """planck black body radiation
        Initiates class
        Inherits the initiation from disc - only now with two extra parameters

        Parameters
        ----------
        r_in : float
            Inner radius of disc - units : Rg.
        r_out : float
            Outer disc radius - units : Rg.
        r_isco : float
            Inner most stable circular orbit (depends on a) - units : Rg.
        inc : float
            System inclination - units : deg.
        mdot : float
            Eddington accretion rate.
        M : float
            BH mass - units : Msol.
        gamma_c : float
            Photon index of resulting Comptonised spectrum.
        kTe_c : float
            Electron temperature (high energy cut-off) of Comptonised medium.
            Units : keV.
        """
        
        Disc.__init__(self, r_in, r_out, a_star, inc, mdot, M, model='AD')
        
        #read new params
        self.gamma_c = gamma_c
        self.kTe_c = kTe_c
    
    
    
    """
    Section for calculating local disc properties. These evolve in time, so 
    will be re-calculated at each step in a light-curve
    """
    
    def _L_local(self, Tseed):
        """
        Calculates the local luminosity for a point/points on the disc
        Ued for normalising the resulting COmptonised spectrum from NTHCOMP
        
        Assumes dL = \sigma_SB T^4
        
        Parameters
        ----------
        Tseed : float OR 2D-array
            Temperature of seed photons at point of calculation.

        Returns
        -------
        L_loc : float OR 2D-array
            Local luminosity of point(s) on disc

        """
        
        L_loc = sigma * Tseed**4
        return L_loc
    
    
    def _T_seed(self, Lirr):
        """
        Calculates seed photon temperature across the disc

        Parameters
        ----------
        Lirr : float OR 2D-array
            Lumnosity of central X-ray source - as seen by point on disc.

        Returns
        -------
        T_seed : 2D-array
            Local seed/disc temperature across disc

        """
        Trep = self.T_rep(Lirr)
        #Applying mask below truncation radius
        Trep = np.ma.masked_where(self.d_mask, Trep)
        Tint = np.ma.masked_where(self.d_mask, self.Td)
        
        T_seed = (Trep**4 + Tint**4)**(1/4)
        return T_seed
    
    
    def Calc_spec_nth(self, Lirr):
        """
        Calculated the spectrum at each point on disc

        Parameters
        ----------
        Lirr : float OR 2D-array
            Lumnosity of central X-ray source - as seen by point on disc.

        Returns
        -------
        Lnu : 1D-array
            Comptonsed disc spectrum

        """
        Es = (self.nu_grid * u.Hz).to(u.keV, equivalencies=u.spectral()).value
        Ts = self._T_seed(Lirr)
        Ts_r = np.mean(Ts, axis=0)
        
        Lloc = self._L_local(Ts)
        Lloc_r = np.trapz(Lloc, self.phi_grid[:, 0], axis=0)
        
        Ts_r_kev = (Ts_r * k_B)/(Ce * 1000) #shifting to kev
        for i in range(len(self.R_grid[0, :])):
            
            if np.isscalar(Ts_r_kev[i]):
                ph_r = donthcomp(Es, [self.gamma_c, self.kTe_c, Ts_r_kev[i], 1, 0])
                normR = Lloc_r[i]/np.trapz(ph_r, Es)
                ph_r = normR * ph_r
                ph_r_nu = (ph_r * u.W/u.keV).to(u.W/u.Hz, 
                                            equivalencies=u.spectral()).value
                if i == 0:
                    fs_r = ph_r_nu
                else:
                    fs_r = np.column_stack((fs_r, ph_r_nu))
            
            else:
                ph_r_nu = np.zeros(len(Es))
                ph_r_nu = np.ma.masked_where(ph_r_nu==0, ph_r_nu)
                
                if i == 0:
                    fs_r = ph_r_nu
                else:
                    fs_r = np.column_stack((fs_r, ph_r_nu))
        
        Lnu = np.trapz(2 * np.cos(self.inc) * fs_r * self.R_grid[0, :], self.R_grid[0, :])
        return Lnu
    
    
    
    def do_evolve_nth(self, l_xs, ts):
        """
        Evolves the Comptonised disc according to an input light_curve

        Parameters
        ----------
        l_xs : 1D-array
            Hard X-ray light_curve - units : F/Fmean.
        ts : 1D-array
            Corresponding time axis - units : days.

        Returns
        -------
        Lnus_all : 2D-array
            Model light-curve for each frequency/energy in nu_grid
            units : W/Hz

        """
        
        Lxs = l_xs * self.Lx #array of x-ray lums
        Lin = np.array([self.Lx]) #array of Ls in play
        
        Lirr = np.ndarray(np.shape(self.tau_grid)) #irradiation array
        for i in tqdm(range(len(ts))):
            
            Lin = np.append(Lin, [Lxs[i]])
            if i == 0:
                t_delay = np.array([np.inf])
            else:
                t_delay += ts[i] - ts[i-1] #Adding time step to delay array
            
            t_delay = np.append(t_delay, [0]) #Appending 0 for current emitted
            
            #Sorting irradiation array
            for j in range(len(t_delay)):
                Lirr[self.tau_grid <= t_delay[j]] = Lin[j]
            
            #Calculating model spec for current time-step
            Lnu_t = self.Calc_spec_nth(Lirr)
            
            if i == 0:
                Lnus_all = Lnu_t
            else:
                Lnus_all = np.column_stack((Lnus_all, Lnu_t))
        
        return Lnus_all
    
    
    
    

            

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import time
    
    r_d = 26.7
    r_out = 400
    a_star = 0
    hx = 10
    inc = 25
    mdot = 10**(-1.3)
    M = 2e8
    gamma_c = 2.5
    kTe_c = 0.2
    
    
    
    """
    Testing calculation speed of new warm-comp model versus old
    """
    
    np.random.seed(123)    
    lxs = np.random.rand(200) + 0.5
    ts = np.linspace(0, 200, 200)
    
    wc_mod = CompDisc(r_d, r_out, a_star, inc, mdot, M, gamma_c, kTe_c)
    nu_v = (1928 * u.AA).to(u.Hz, equivalencies=u.spectral()).value
    

    print('evolving nthcomp model')
    t_nth_1 = time.time()
    Ls_nth = wc_mod.do_evolve_nth(lxs, ts)
    t_nth_2 = time.time()
    print('Runtime = {} mins'.format((t_nth_2 - t_nth_1)/60))
    
    idx_nu_nth = np.abs(wc_mod.nu_grid - nu_v).argmin()
    wc_nth_nu = Ls_nth[idx_nu_nth, :]
    
    plt.plot(ts, lxs)
    plt.plot(ts, wc_nth_nu/np.mean(wc_nth_nu))
    plt.show()
    

    
    