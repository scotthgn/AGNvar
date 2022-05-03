#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  1 10:56:36 2022

@author: wljw75
"""

"""
Module to define an AGN geometry - and then calculate spectra, impulse response
functions, and model light-curves (based off an input HX light-curve).

Uses the models defined within the model_bin/ directory - this is more of a 
user end to make combining the models easier... If you want to play around
with the models directly, then have a look in the model_bin!
"""


import numpy as np
import astropy.units as u
import astropy.constants as const

from scipy.integrate import quad
from pyNTHCOMP import donthcomp

import warnings


#Stop all the run-time warnings (we know why they happen - doesn't affect the output!)
warnings.filterwarnings('ignore') 

"""
Constants
"""
G = (const.G * const.M_sun).value #Grav const in units m^3 s^-1 Msol^-1
c = const.c.value #speed of light, m/s
h = const.h.value #planck constant, Js
sigma_T = const.sigma_T.value #Thompson cross section, m^2
sigma_sb = const.sigma_sb.value #Stefan-Boltzmann const, W m^-2 K^-4
m_p = const.m_p.value #Proton mass, kg
k_B = const.k_B.value #Boltzmann const, J K^-1





class AGN:
    """
    Main AGN object - Defines all the general properties we expect to be the 
    same across models. e.g efficiency, eddington luminosity, etc.
    
    Also deals with all the 'book-keeping' with regards to defining 
    energy and radial grids, and dealing with units
    """
    
    Emin = 1e-4
    Emax = 1e4
    numE = 500
    mu = 0.55 #mean particle mass - fixed at solar abundances
    A = 0.3 #Disc albedo = fixed at 0.3 for now
    
    dr_dex = 40 #radial grid spacing - N points per decade
    dphi = 0.01 #azimuthal grid spacing
    
    return_disc = True #flags for what components to return
    return_warm = True #Determined by the swithcing parameters
    return_hot = True
    
    units = 'SI' #Default units to be returned
    as_flux = False
    
    def __init__(self,
                 M,
                 dist,
                 log_mdot,
                 astar,
                 cosi,
                 z):
        """
        Initiates the AGN object 

        Parameters
        ----------
        M : float
            BH mass - units : Msol.
        dist : float
            Co-Moving distance - units : Mpc
        log_mdot : float
            log of mass accretion rate - units : L/Ledd
        astar : float
            Dimensionless spin parameter
        cosi : float
            cosine of inclination
        z : float
            Redshift (not currently doing anything - will be updated later)
        """
        
        #Read parameters
        self.M = M 
        self.D, self.d = dist, (dist * u.Mpc).to(u.cm).value
        self.mdot = 10**(log_mdot)
        self.a = astar
        self.cosinc = cosi
        self.inc = np.arccos(cosi)
        self.z = z
        
        #Calculating disc params 
        self._calc_risco()
        self._calc_r_selfGravity()
        self._calc_Ledd()
        self._calc_efficiency()
        
        #physical conversion factors
        self.Mdot_edd = self.L_edd/(self.eta * c**2)
        self.Rg = (G * self.M)/(c**2)
        self._calc_Dl()
        
        
        #Creating azimuthal bins
        self.phis = np.arange(0, 2*np.pi + self.dphi, self.dphi)
        
        #Energy/frequency grid
        self.Egrid = np.geomspace(self.Emin, self.Emax, self.numE)
        self.nu_grid = (self.Egrid * u.keV).to(u.Hz,
                                equivalencies=u.spectral()).value
        
        

    
    
    
    ##########################################################################
    #---- Units and energy grids
    ##########################################################################
    
    def set_cgs(self):
        """
        Changes output spectra to cgs units

        """
        self.units = 'cgs'
    
    def set_SI(self):
        """
        changes to si units - NOTE - these are default!!

        """
        self.units = 'SI'
        
    def set_counts(self):
        """
        Changes output spectra to photons/s/keV

        """
        self.units = 'counts'
    
    def set_flux(self):
        """
        Changes output spectra from luminosity to flux

        """
        self.as_flux = True
    

    
    def _new_units(self, Lnus):
        """
        Converts to whatever unit is currently set
        Always from SI, as this is what the calculations are done in

        """
        if self.units == 'cgs':
            Lnus = Lnus*1e7
        
        elif self.units == 'counts':
            flxs = (Lnus * u.W/u.Hz).to(u.keV/u.s/u.keV,
                                                equivalencies=u.spectral()).value
            
            if np.ndim(flxs) == 1:
                Lnus = flxs/self.Egrid
            else:
                Lnus = flxs/self.Egrid[:, np.newaxis]
        
        elif self.units == 'SI':
            pass
        
        
        if self.as_flux == True:
            Lnus = self._to_flux(Lnus)
        
        return Lnus
        
    
    def _to_flux(self, Lnus):
        """
        Converts luminosity to flux. Either per cm^2 or per m^2, depending
        on if SI or cgs

        """
        if self.units.__contains__('cgs') or self.units.__contains__('counts'):
            dist = self.dl
        else:
            dist = self.dl/100
        
        return Lnus/(4*np.pi*dist**2)
    
    
    def new_ear(self, ear):
        """
        Defines new energy grid if necessary

        Parameters
        ----------
        ear : 1D-array
            New energy grid - units : keV.

        """
        self.Egrid = ear         
        self.nu_grid = (self.Egrid * u.keV).to(u.Hz,
                                equivalencies=u.spectral()).value
        self.nu_obs = self.nu_grid/(1 + self.z) #Observers frame
        
        self.Emin = min(self.Egrid)
        self.Emax = max(self.Egrid)
        self.numE = len(self.Egrid)

    
    
    ##########################################################################
    #---- Disc properties
    ##########################################################################
    
    def _calc_Ledd(self):
        """
        Caclulate eddington Luminosity

        """
        #Ledd = (4 * np.pi * G * self.M * m_p * c)/sigma_T
        Ledd = 1.39e31 * self.M 
        #print(Ledd, Ledd_c)
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

        self.risco = 3 + Z2 - np.sign(self.a) * np.sqrt(
            (3 - Z1) * (3 + Z1 + 2*Z2))
    
    
    def _calc_r_selfGravity(self):
        """
        Calcultes the self gravity radius according to Laor & Netzer 1989
        
        NOTE: Assuming that \alpha=0.1 - in future should figure out how to
        constrain this properly!!!

        """
        alpha = 0.1 #assuming turbulence NOT comparable to sound speed
        #See Laor & Netzer 1989 for more details on constraining this parameter
        m9 = self.M/1e9
        self.r_sg = 2150 * m9**(-2/9) * self.mdot**(4/9) * alpha**(2/9)
    
    
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
        
        self.eta = 1 - np.sqrt(1 - 2/(3*self.risco))
    
    
    def _calc_NTparams(self, r):
        """
        Calculates the Novikov-Thorne relativistic factors.
        see Active Galactic Nuclei, J. H. Krolik, p.151-154
        and Page & Thorne (1974)

        """
        y = np.sqrt(r)
        y_isc = np.sqrt(self.risco)
        y1 = 2 * np.cos((1/3) * np.arccos(self.a) - (np.pi/3))
        y2 = 2 * np.cos((1/3) * np.arccos(self.a) + (np.pi/3))
        y3 = -2 * np.cos((1/3) * np.arccos(self.a))

        
        B = 1 - (3/r) + ((2 * self.a)/(r**(3/2)))
        
        C1 = 1 - (y_isc/y) - ((3 * self.a)/(2 * y)) * np.log(y/y_isc)
        
        C2 = ((3 * (y1 - self.a)**2)/(y*y1 * (y1 - y2) * (y1 - y3))) * np.log(
            (y - y1)/(y_isc - y1))
        C2 += ((3 * (y2 - self.a)**2)/(y*y2 * (y2 - y1) * (y2 - y3))) * np.log(
            (y - y2)/(y_isc - y2))
        C2 += ((3 * (y3 - self.a)**2)/(y*y3 * (y3 - y1) * (y3 - y2))) * np.log(
            (y - y3)/(y_isc - y3))
        
        C = C1 - C2
        
        return C/B
        
        
    
    
    def calc_Tnt(self, r):
        """
        Calculates Novikov-Thorne disc temperature^4 at radius r. 
 
        """
        Rt = self._calc_NTparams(r)
        const_fac = (3 * G * self.M * self.mdot * self.Mdot_edd)/(
            8 * np.pi * sigma_sb * (r * self.Rg)**3)
        
        T4 = const_fac * Rt

        return T4
    
    
    def calc_Trep(self, r, Lx_t):
        """
        Calculates current re-processed temperature
        
        Parameters
        ----------
        r : float OR array
            Radial coordinate - units : Rg
        Lx_t : float OR array
            X-ray luminosity seen at point r, phi at time t - units : W
        """
        R = r * self.Rg
        H = self.hmax * self.Rg
        
        Frep = (0.5*Lx_t)/(4*np.pi * (R**2 + H**2))
        Frep *= H/np.sqrt(R**2 + H**2)
        Frep *= (1 - self.A) 
        
        T4rep = Frep/sigma_sb
        
        #T4rep = (Lx_t * H)/(4*np.pi * sigma_sb * 
        #                          (R**2 + H**2)**(3/2))
        return T4rep * (1 - self.A)
    
    
    def calc_Ttot(self, r, Lx_t):
        """
        Caclculates total temperature
        Re-processing always on...
        """
        T4tot = self.calc_Tnt(r) + self.calc_Trep(r, Lx_t)
 
        return T4tot
    
    
    def _calc_Dl(self):
        """
        Calculates luminosity distance to source
        """
        
        self.Dl = self.D * (1+self.z) #In Mpc
        self.dl = self.d * (1+self.z) #In cm
    
    
    def delay_surf(self, r, phi):
        """
        Calculates time-delay at point r, phi - mostly following 
        Welsh & Horne 1991, but with addition of coronal height

        Parameters
        ----------
        r : float OR array
            Radial coordinate - units : Rg.
        phi : float OR array
            Azimuthal coordinate - units : rad.

        Returns
        -------
        tau : float OR array
            Time delay - units : days.

        """
        R = r * self.Rg
        Hx = self.hmax * self.Rg
        #Delay surface in seconds
        tau_sec = (1/c) * (np.sqrt(R**2 + Hx**2) + Hx*np.cos(self.inc) - 
                           R*np.cos(phi)*np.sin(self.inc)) 
    
        tau = tau_sec/(60*60*24) # Delay surface in days
        return tau
    
    

    def _make_rbins(self, logr_in, logr_out):
        """
        Creates an array of radial bin edges, with spacing defined by dr_dex
        Calculates the bin edges from r_out and down to r_in. IF the bin
        between r_in and r_in+dr is less than dlog_r defined by dr_dex, then
        we simply create a slightly wider bin at this point to accomodate
        for the difference

        Parameters
        ----------
        logr_in : float
            Inner radius of model section - units : Rg.
        logr_out : float
            Outer radius of model section - units : Rg.

        Returns
        -------
        logr_bins : 1D-array
            Radial bin edges for section - units : Rg.

        """
        i = logr_out
        logr_bins = np.array([np.float64(logr_out)]) 
        while i > logr_in:
            r_next_edge = i - self.dlog_r
            logr_bins = np.insert(logr_bins, 0, r_next_edge)
            i = r_next_edge

       
        if logr_bins[0] != logr_in:
            if logr_bins[0] < logr_in:
                if len(logr_bins) > 1:
                    logr_bins = np.delete(logr_bins, 0)
                    logr_bins[0] = logr_in
                else:
                    logr_bins[0] = logr_in
            else:
                logr_bins[0] = logr_in
        
        return logr_bins
    
    
    
    def bb_radiance_ann(self, Ts):
        """
        Black-body radiance across the annulus

        Parameters
        ----------
        Ts : float
            Mean temperature at annulus - units : K.

        Returns
        -------
        \pi*Bnu : D-array - shape = len(self.nu_grid)
            Black body spectrum for each azimuthal coord.

        """
        pre_fac = (2 * h * self.nu_grid**3)/(c**2)
        exp_fac = np.exp((h * self.nu_grid)/(k_B * Ts)) - 1
        Bnu = pre_fac / exp_fac
        
        return np.pi * Bnu



##############################################################################
#### The main Composite models
##############################################################################
    
    
class AGNsed_var(AGN):
    
    """
    This is essentially the same as AGNSED in XSPEC (Kubota & Done 2018),
    however evolves it through time so you can do spectral timing analysis
    """
    
    return_disc = True #flags for what components to return
    return_warm = True #Determined by the swithcing parameters
    return_hot = True
    
    def __init__(self,
                 M,
                 dist,
                 log_mdot,
                 astar,
                 cosi,
                 kTe_h,
                 kTe_w,
                 gamma_h,
                 gamma_w,
                 r_h,
                 r_w,
                 log_rout,
                 hmax,
                 z):
        """
        Initiates the AGNsed object - defines geometry

        Parameters
        ----------
        M : float
            BH mass - units : Msol.
        dist : float
            Co-Moving distance - units : Mpc
        log_mdot : float
            log of mass accretion rate - units : L/Ledd
        astar : float
            Dimensionless spin parameter
        cosi : float
            cosine of inclination
        kTe_h : float
            Electron temperature for hot corona (high energy rollover)
            Units : keV
        kTe_w : float
            Electron temperature for warm corona (high energy rollover)
            Units : keV
        gamma_h : float
            Hot Compton photon index
        gamma_w : float
            Warm Compton photon index
        r_h : float
            Outer edge of hot corona (or inner edge of warm..)
            If -ve uses risco
            Units : Rg
        r_w : float 
            Outer edge of warm corona (or inner edge of standard disc)
            If -ve uses risco
            Units : Rg
        log_rout : float
            Outer edge of disc - units : Rg
            If -ve uses r_sg
        hmax : float
            Max height of hot corona - units : Rg
        z : float
            Redshift (not currently doing anything - will be updated later)
        """
        
        #getting properties definied in __init__ from AGN parent class
        super().__init__(M, dist, log_mdot, astar, cosi, z) 
        
        #Read remaining parameters
        self.kTe_h = kTe_h
        self.kTe_w = kTe_w
        self.gamma_h = gamma_h
        self.gamma_w = gamma_w
        self.r_h = r_h
        self.r_w = r_w
        self.r_out = 10**(log_rout)
        self.hmax = hmax
        
        
        if log_rout < 0:
            self.r_out = self.r_sg
        
        if r_h < 0:
            self.r_h = self.risco
        
        if r_w < 0:
            self.r_w = self.risco
        
        
        #Creating radial grid for each component
        self.dlog_r = 1/self.dr_dex
        self.logr_ad_bins = self._make_rbins(np.log10(self.r_w), np.log10(self.r_out))
        self.logr_wc_bins = self._make_rbins(np.log10(self.r_h), np.log10(self.r_w))
        
        #If too narrow to create a bin with correct size, just set one bin
        #instead
        if len(self.logr_ad_bins) == 1:
            self.logr_ad_bins = np.array([np.log10(self.r_w), np.log10(self.r_out)])
        
        if len(self.logr_wc_bins) == 1:
            self.logr_wc_bins = np.array([np.log10(self.r_h), np.log10(self.r_w)])
            
        
        #Creating delay surface for warm region and disc
        rd_mids = 10**(self.logr_ad_bins[:-1] + self.dlog_r/2)
        rw_mids = 10**(self.logr_wc_bins[:-1] + self.dlog_r/2)
        
        rad_mesh, phi_admesh = np.meshgrid(rd_mids, self.phis)
        rwc_mesh, phi_wcmesh = np.meshgrid(rw_mids, self.phis)
        
        self.tau_ad = self.delay_surf(rad_mesh, phi_admesh)
        self.tau_wc = self.delay_surf(rwc_mesh, phi_wcmesh)
        
        #Mean X-ray luminosity
        self.Lx = self.hotCorona_lumin()
        
    
    
    ##########################################################################
    #### Calculate spectra
    ##########################################################################
    
    def disc_annuli(self, r, dr, Lx_t):
        """
        Calculates total spectrum for annulus at r, width dr
        Does this by calculating mean temperature around annulus, and using
        that to calculate spectrum

        Parameters
        ----------
        r : float
            Annulus mid-point - units : Rg.
        dr : float
            Annulus width - unbits : Rg.
        Lx_t : 1D-array - shape = shape(self.phis)
            X-ray luminosity seen at each point around the annulus at time t.

        Returns
        -------
        Lnu_ann : 1D-array - shape = shape(self.nu_grid)
            Spectrum from this annulus

        """
        
        T4ann = self.calc_Ttot(r, Lx_t)
        Tann_mean = np.mean(T4ann**(1/4))
        bb_ann = self.bb_radiance_ann(Tann_mean)

        Lnu_ann  = 4*np.pi*r*dr * self.Rg**2 * bb_ann #multiplying by dA to get actual normalisation
        return Lnu_ann

        
    def disc_spec_t(self, Lx_t):
        """
        Calculates total disc spectrum, given an incident luminosity Lx(t)

        Parameters
        ----------
        Lx_t : float OR 2D-array - shape = [N_rbin - 1, N_phi]
            Incident luminosity seen at each point on disc at time t.
            IF float then assume constant irradiation across disc
            Units : W

        """
        for i in range(len(self.logr_ad_bins) - 1):
            dr_bin = 10**self.logr_ad_bins[i+1] - 10**self.logr_ad_bins[i]
            rmid = 10**(self.logr_ad_bins[i] + self.dlog_r/2)
            
            if np.ndim(Lx_t) == 0:
                Lx_r = Lx_t
            else:
                Lx_r = Lx_t[:, i]
                
            Lnu_r = self.disc_annuli(rmid, dr_bin, Lx_r)
            
            if i == 0:
                Lnu_all = Lnu_r
            else:
                Lnu_all = np.column_stack((Lnu_all, Lnu_r))
        
        if np.shape(Lnu_all) != np.shape(self.Egrid):
            Lnu_tot = np.sum(Lnu_all, axis=-1)
        else:
            Lnu_tot = Lnu_all
        
        return Lnu_tot *self.cosinc/0.5
    
    
    
    """
    Now doing warm Compton region
    """
    def warm_annuli(self, r, dr, Lx_t):
        """
        Calculates spectrum from annulus in warm comp region
        Due to performance issues - here we calculate mean temperature around
        annulus, and use that to calculate the spectrum/normalisation.
        
        Note, this model uses nthcomp, which assumes an underlying black
        body spectrum - which makes sense it it is a disc. We assume seed
        photon temperature given by disc temperature...

        Parameters
        ----------
        r : float
            Mid point of annulus - units : Rg.
        dr : float
            Width of annulus - units : Rg.
        Lx_t : float OR 1D-array
            X-ray luminosity seen at annulus.

        Returns
        -------
        Lnu_ann : 1D-array
            Spectrum at annulus.

        """
        T4ann = self.calc_Ttot(r, Lx_t)
        Tann_mean = np.mean(T4ann**(1/4))
        
        kTann = k_B * Tann_mean
        kTann = (kTann * u.J).to(u.keV).value #converting T to keV for nthcomp
        
        ph_nth = donthcomp(self.Egrid, [self.gamma_w, self.kTe_w,
                                        kTann, 1, 0])
        ph_nth = (ph_nth * u.W/u.keV).to(u.W/u.Hz, 
                                            equivalencies=u.spectral()).value
        
        norm = sigma_sb * (Tann_mean**4) * 4*np.pi*r*dr * self.Rg**2
        radiance = np.trapz(ph_nth, self.nu_grid)
        if radiance == 0:
            Lnu_ann = np.zeros(len(self.nu_grid))
        else:
            Lnu_ann = norm * (ph_nth/radiance)
        
        return Lnu_ann
    
    
    def warm_spec_t(self, Lx_t):
        """
        Calculates total warm spectrum, given an incident luminosity Lx(t)

        Parameters
        ----------
        Lx_t : float OR 2D-array - shape = [N_rbin - 1, N_phi]
            Incident luminosity seen at each point on disc at time t.
            IF float then assume constant irradiation across disc
            Units : W

        """
        for i in range(len(self.logr_wc_bins) - 1):
            dr_bin = 10**self.logr_wc_bins[i+1] - 10**self.logr_wc_bins[i]
            rmid = 10**(self.logr_wc_bins[i] + self.dlog_r/2)
            
            if np.ndim(Lx_t) == 0:
                Lx_r = Lx_t
            else:
                Lx_r = Lx_t[:, i]
                
            Lnu_r = self.warm_annuli(rmid, dr_bin, Lx_r)
            
            if i == 0:
                Lnu_all = Lnu_r
            else:
                Lnu_all = np.column_stack((Lnu_all, Lnu_r))
        
        if np.shape(Lnu_all) != np.shape(self.Egrid):
            Lnu_tot = np.sum(Lnu_all, axis=-1)
        else:
            Lnu_tot = Lnu_all
        
        return Lnu_tot * self.cosinc/0.5
    
    
    
    """
    Section for hot Compton component
    Note - it is also here where mean X-ray luminosity is calculated
    """
    def seed_tempHot(self):
        """
        Calculated seed photon temperature for the hot compton region.
        Follows xspec model agnsed, from Kubota & Done (2018).
        For simplicity only including temperature due to gravitational
        dissipation - not reprocessing. Will update this later
        
        Returns
        -------
        kT_seed : float
            Seed photon temperature for hot compton - units : keV

        """
        T4_edge = self.calc_Tnt(self.r_h) #inner disc T in K
        Tedge = T4_edge**(1/4)
        
        kT_edge = k_B * Tedge #units J
        kT_edge = (kT_edge * u.J).to(u.keV).value
        if self.r_w != self.r_h:
            #If there is a warm compton region then seed photons mostly from
            #here. Will then need to include Compton y-param
            ysb = (self.gamma_w * (4/9))**(-4.5)
            kT_seed = np.exp(ysb) * kT_edge
            
        else:
            #If only disc down to r_hot seed temp will be same as inner disc temp
            kT_seed = kT_edge

        return kT_seed
    
    
    def Lseed_hotCorona(self):
        """
        Calculated luminsoty of seed photons emitted at radius r, intercepted
        by corona

        Returns
        -------
        Lseed_tot : float
            Total seed photon luminosity seen by corona - units : W

        """
        logr_all_grid = self._make_rbins(np.log10(self.r_h), np.log10(self.r_out))
        Lseed_tot = 0
        for i in range(len(logr_all_grid) - 1):
            dr = 10**logr_all_grid[i+1] - 10**logr_all_grid[i]
            rmid = 10**(logr_all_grid[i] + 0.5 * self.dlog_r)
            
            if self.hmax <= rmid:
                theta_0 = np.arcsin(self.hmax/rmid)
                cov_frac = theta_0 - 0.5 * np.sin(2*theta_0)
            else:
                cov_frac = 1
            
            T4_ann = self.calc_Tnt(rmid)
            
            Fr = sigma_sb * T4_ann
            Lr = 2 * 2*np.pi*rmid*dr * Fr * cov_frac/np.pi * self.Rg**2


            Lseed_tot += Lr
        
        return Lseed_tot
    
    
    def hotCorona_lumin(self):
        """
        Calculates the coronal luminosity - used as normalisaton for the 
        hot compton spectral component AND as the mean X-ray luminosity.
        
        Calculated as Lhot = Ldiss + Lseed
        
        where Ldiss is the energy dissipated from the accretion flow, and
        Lseed is the seed photon luminosity intercpted by the corona

        """
        
        Ldiss, err = quad(lambda rc: 2*sigma_sb*self.calc_Tnt(rc) * 2*np.pi*rc * self.Rg**2,
                     self.risco, self.r_h)

        Lseed = self.Lseed_hotCorona()
        
        Lhot = Ldiss + Lseed
        return Lhot
    
    
    def hot_spec(self):
        """
        Calculates spectrum of hot comptonised region - no relativity

        """
        kTseed = self.seed_tempHot()
        Lum = self.hotCorona_lumin()
        
        if kTseed < self.kTe_h:
            ph_hot = donthcomp(self.Egrid, [self.gamma_h, self.kTe_h, kTseed, 1, 0])
            ph_hot = (ph_hot * u.W/u.keV).to(u.W/u.Hz, 
                                            equivalencies=u.spectral()).value
        
            Lnu_hot = Lum * (ph_hot/np.trapz(ph_hot, self.nu_grid))
        
        else:
            Lnu_hot = np.zeros(len(self.Egrid))
            
        return Lnu_hot
    
    

    
    ##########################################################################
    #---- Spectral evolution
    ##########################################################################
    
    def evolve_spec(self, lxs, ts):
        """
        Evolves the spectral model according to an input x-ray light-curve

        Parameters
        ----------
        lxs : 1D-array
            X-ray light-curve.
        ts : 1D-array
            Corresponding time stamps.
            
        """
        #First checking that beginning of time array is 0!
        if ts[0] != 0:
            ts = ts - ts[0]
        
        #Now checking that light-curve flux is fractional
        if np.mean(lxs) != 1:
            lxs = lxs/np.mean(lxs)
        
        #getting mean hot spec - as this just goes up and down...
        #no time delay for this component...
        if self.return_hot == True:
            Lh_mean = self.hot_spec()
        else:
            Lh_mean = np.zeros(len(self.nu_grid))
        
        #Now evolving light-curves!
        Lxs = lxs * self.Lx #array of x-ray lums
        Lin = np.array([self.Lx]) #array of Ls in play
        
        Lirr_ad = np.ndarray(np.shape(self.tau_ad))
        Lirr_wc = np.ndarray(np.shape(self.tau_wc))
        for j in range(len(ts)):

            Lin = np.append(Lin, [Lxs[j]])
            if j == 0:
                t_delay = np.array([np.inf])
            else:
                t_delay += ts[j] - ts[j-1] #Adding time step to delay array
            
            t_delay = np.append(t_delay, [0]) #Appending 0 for current emitted
            
            #Sorting irradiation arrays
            for k in range(len(t_delay)):
                Lirr_ad[self.tau_ad <= t_delay[k]] = Lin[k]
                Lirr_wc[self.tau_wc <= t_delay[k]] = Lin[k]
            
            #Evolving spectral components
            if self.return_disc == True and len(self.logr_ad_bins) > 1:
                Ld_t = self.disc_spec_t(Lirr_ad)
            else:
                Ld_t = np.zeros(len(self.nu_grid))
            
            if self.return_warm == True and len(self.logr_wc_bins) > 1:
                Lw_t = self.warm_spec_t(Lirr_wc)
            else:
                Lw_t = np.zeros(len(self.nu_grid))
            
            Lh_t = Lh_mean * lxs[j]
            Ltot_t = Ld_t + Lw_t + Lh_t
            

            if j == 0:
                Ld_all = Ld_t
                Lw_all = Lw_t
                Lh_all = Lh_t
                Ltot_all = Ltot_t
            
            else:
                Ld_all = np.column_stack((Ld_all, Ld_t))
                Lw_all = np.column_stack((Lw_all, Lw_t))
                Lh_all = np.column_stack((Lh_all, Lh_t))
                Ltot_all = np.column_stack((Ltot_all, Ltot_t))
        
        
        self.Ld_t_all = self._new_units(Ld_all)
        self.Lw_t_all = self._new_units(Lw_all)
        self.Lh_t_all = self._new_units(Lh_all)
        self.Ltot_t_all = self._new_units(Ltot_all)
        return self.Ltot_t_all
    
    
    
    def mean_spec(self):
        """
        Calculated the mean spectrum + each component
        Uses X-ray luminosity derived from model for irradiation
        """
        
        #disc component
        #including condition on r_bin to avoid resolution error
        #if you absoltely want to see tiny contributions then increase radial resolution
        if self.return_disc == True and len(self.logr_ad_bins) > 1:
            Lnu_d = self.disc_spec_t(self.Lx)
        else:
            Lnu_d = np.zeros(len(self.nu_grid))
        
        #warm component
        if self.return_warm == True and len(self.logr_wc_bins) > 1:
            Lnu_w = self.warm_spec_t(self.Lx)
        else:
            Lnu_w = np.zeros(len(self.nu_grid))
        
        if self.return_hot == True and self.r_h != self.risco:
            Lnu_h = self.hot_spec()
        else:
            Lnu_h = np.zeros(len(self.nu_grid))
        
        Ltot = Lnu_d + Lnu_w + Lnu_h
        
        self.Lnu_d = self._new_units(Lnu_d)
        self.Lnu_w = self._new_units(Lnu_w)
        self.Lnu_h = self._new_units(Lnu_h)
        self.Lnu_tot = self._new_units(Ltot)
        return self.Lnu_tot
        
            
    def generate_lightcurve(self, band, band_width, lxs=None, ts=None):
        """
        Generated a light-curve for a band centered on nu, with bandwidth dnu
        
        Currently only does top-hat response/bandpass. Might be updated later.
        
        Input MUST be in whatever units you have set units to be!!!
        i.e if cgs of SI - then Hz,
            if counts - then keV
        IF you have NOT set any units - then default is currently SI, and you
        need to pass band in Hz
        
        NOTE: If band_width smaller than bin-width in model grid, then model bin width
        used instead!
        Parameters
        ----------
        band : float
            Midpoint in bandpass - units : Hz OR keV.
        band_width : float
            Bandwidth - units : Hz or keV.
        lxs : 1D-array, OPTIONAL
            X-ray light-curve - only needed if evolved spec NOT already calculated
        ts : 1D-array, OPTIONAL
            Light-curve time stamps

        """
        
        if hasattr(self, 'Ltot_t_all'):
            Ltot_all = self.Ltot_t_all
        else:
            if lxs == None:
                raise ValueError('NONE type light-curve not permitted!! \n'
                                 'Either run evolve_spec() FIRST \n'
                                 'OR pass a light-curve here!')
            else:    
                Ltot_all = self.evolve_spec(lxs, ts)
        
        
        if self.units == 'SI' or self.units == 'cgs':
            idx_mod_up = np.abs(band + band_width/2 - self.nu_grid).argmin()
            idx_mod_low = np.abs(band - band_width/2 - self.nu_grid).argmin()
            
            if idx_mod_up == idx_mod_low:
                Lcurve = Ltot_all[idx_mod_up, :] * band_width
                
            else:
                Lc_band = Ltot_all[idx_mod_low:idx_mod_up+1, :]
                Lcurve = np.trapz(Lc_band, self.nu_grid[idx_mod_low:idx_mod_up+1], axis=0)
            
        elif self.units == 'counts':
            idx_mod_up = np.abs(band + band_width/2 - self.Egrid).argmin()
            idx_mod_low = np.abs(band - band_width/2 - self.Egrid).argmin()
            
            if idx_mod_up == idx_mod_low:
                Lcurve = Ltot_all[idx_mod_up, :] * band_width
            
            else:
                Lc_band = Ltot_all[idx_mod_low:idx_mod_up+1, :]
                Lcurve = np.trapz(Lc_band, self.Egrid[idx_mod_low:idx_mod_up+1], axis=0)
        
        return Lcurve
    
    
    
    
    """
    Section for calculating impulse response functions
    """
    
    def response_components(self, band, band_width):
        """
        Calculates the response for each disc component for a band
        with bad_width + the total responese, to a 'delta' function flash in
        the x-ray.
        
        This is to get an idea of how different parts of the accretion flow
        respond at a given time

        Parameters
        ----------
        band : float
            Band central frequency/energy - units : Hz/keV.
        band_width : float
            Band width - units : Hz/keV.

        Returns
        -------
        irf_comp : dict
            Dictionary containing impulse response functions for each model
            component

        """
        tau_max = self.delay_surf(self.r_out, np.pi)
        self.t_imp = np.linspace(0, tau_max, 200) #Time array for impulse L-curve
        Tg = (self.Rg/c)/(24*3600) #Light travel time for 1 Rg in days
        dt = 10*Tg #flash duration

        x_imp = np.full(len(self.t_imp), 1)
        idx_max = np.abs(dt - self.t_imp).argmin()
        x_imp[0:idx_max+1] = 2

        
        Lrp_tot = self.evolve_spec(x_imp, self.t_imp)
        L_imp = self.generate_lightcurve(band, band_width)
        L_int_all = self.mean_spec()
        
        if self.units == 'cgs' or self.units == 'SI':
            idx_mod_up = np.abs(band + band_width/2 - self.nu_grid).argmin()
            idx_mod_low = np.abs(band - band_width/2 - self.nu_grid).argmin()
            
            if idx_mod_up == idx_mod_low:
                L_int = L_int_all[idx_mod_up] * band_width
            
            else:
                Lint_band = L_int_all[idx_mod_low:idx_mod_up+1]
                L_int = np.trapz(Lint_band, self.nu_grid[idx_mod_low:idx_mod_up+1], axis=0)
        
        elif self.units == 'counts':
            idx_mod_up = np.abs(band + band_width/2 - self.Egrid).argmin()
            idx_mod_low = np.abs(band - band_width/2 - self.Egrid).argmin()
            
            if idx_mod_up == idx_mod_low:
                L_int = L_int_all[idx_mod_up, :] * band_width
            
            else:
                Lint_band = L_int_all[idx_mod_low:idx_mod_up+1]
                L_int = np.trapz(Lint_band, self.Egrid[idx_mod_low:idx_mod_up+1], axis=0)
                
        #irf_mod = (L_imp/L_int - 1)/(2*dt)  
        tot_resp = (L_imp/L_int - 1)/(2*dt)
        
        
        return tot_resp
    
    
    def radial_transfer(self, r, dr):
        
        tau_min = self.delay_surf(r, 0)
        tau_max = self.delay_surf(r, np.pi)
        Tg = ((self.Rg*dr)/c)/(24*3600) #Light travel time for annulus in days
        dt = Tg/4 #flash duration
        delays = np.linspace(0, 8, 1000)
        
        def delta(phi, tau):
            tau_ann = self.delay_surf(r, phi)
            dirac_gauss = np.exp(-((tau - tau_ann)**2)/(dt**2))
            return dirac_gauss
            
        T = np.array([])
        for i in range(len(delays)):
            T_tau = quad(delta, 0, 2*np.pi, args=(delays[i]))[0]
            T = np.append(T, [T_tau/(2*np.pi)])
        
        norm = 1/np.trapz(T, delays)
        T *= norm
        return T, delays



class AGNdark_var(AGN):
    
    """
    A model where you have a standard disc extending to risco. However below
    some darkeining radius, r_d, all accretion power is transported to the corona
    (probably through magnetic fields or something...)
    Hence, in the region r_d to risco, you only see the contribution due to 
    re-processing
    """
    
    return_AD = True #Flags for neglecting/including components
    return_DD = True
    return_corona = True
    
    def __init__(self,
                 M,
                 dist,
                 log_mdot,
                 astar,
                 cosi,
                 kTe_c,
                 gamma_c,
                 r_d,
                 log_rout,
                 hmax,
                 z):
        """
        Initiates AGNdark object

        Parameters
        ----------
        M : float
            BH mass - units : Msol.
        dist : float
            Co-Moving distance - units : Mpc
        log_mdot : float
            log of mass accretion rate - units : L/Ledd
        astar : float
            Dimensionless spin parameter
        cosi : float
            cosine of inclination
        kTe_c : float
            Electron temperature for corona (high energy rollover)
            Units : keV
        gamma_c : float
            Corona photon index
        r_d : float
            Disc darkening radius
            If -ve uses risco
            Units : Rg
        log_rout : float
            Outer edge of disc - units : Rg
            If -ve uses r_sg
        hmax : float
            Max height of hot corona - units : Rg
        z : float
            Redshift (not currently doing anything - will be updated later)
        """
        
        #getting properties definied in __init__ from AGN parent class
        super().__init__(M, dist, log_mdot, astar, cosi, z)
        
        #Reading remaining parameters
        self.kTe_c = kTe_c
        self.gamma_c = gamma_c
        self.r_d = r_d
        self.r_out = 10**(log_rout)
        self.hmax = hmax
        
        
        if log_rout < 0:
            self.r_out = self.r_sg
        
        if r_d < 0:
            self.r_d = self.risco
        
        
        #Creating radial grid for each component
        self.dlog_r = 1/self.dr_dex
        self.logr_ad_bins = self._make_rbins(np.log10(self.r_d), np.log10(self.r_out))
        self.logr_dd_bins = self._make_rbins(np.log10(self.risco), np.log10(self.r_d))
        
        #If too narrow to create a bin with correct size, just set one bin
        #instead
        if len(self.logr_ad_bins) == 1:
            self.logr_ad_bins = np.array([np.log10(self.r_d), np.log10(self.r_out)])
        
        if len(self.logr_dd_bins) == 1:
            self.logr_dd_bins = np.array([np.log10(self.risco), np.log10(self.r_d)])
        
        
        #Creating delay surface for dark region and disc
        rad_mids = 10**(self.logr_ad_bins[:-1] + self.dlog_r/2)
        rdd_mids = 10**(self.logr_dd_bins[:-1] + self.dlog_r/2)
        
        rad_mesh, phi_admesh = np.meshgrid(rad_mids, self.phis)
        rdd_mesh, phi_ddmesh = np.meshgrid(rdd_mids, self.phis)
        
        self.tau_ad = self.delay_surf(rad_mesh, phi_admesh)
        self.tau_dd = self.delay_surf(rdd_mesh, phi_ddmesh)
        
        #X-ray luminosity
        self.Lx = self.Corona_lumin()
    
    
    
    ##########################################################################
    #### Calculate spectra
    ##########################################################################
    
    def AD_annuli(self, r, dr, Lx_t):
        """
        Calculates contribution from standard disc annulus as time t

        Parameters
        ----------
        r : float
            Annulus midpoint - units : Rg.
        dr : float
            Annulus width - units : Rg.
        Lx_t : 1D-array - shape = shape(self.phis)
            X-ray luminosity seen at each point around the annulus at time t.

        """
        T4ann = self.calc_Ttot(r, Lx_t)
        Tann_mean = np.mean(T4ann**(1/4))
        bb_ann = self.bb_radiance_ann(Tann_mean)

        Lnu_ann  = 4*np.pi*r*dr * self.Rg**2 * bb_ann #multiplying by dA to get actual normalisation
        return Lnu_ann
    
    
    def AD_spec_t(self, Lx_t):
        """
        Calculates total standard disc spectrum, given an incident luminosity Lx(t)

        Parameters
        ----------
        Lx_t : float OR 2D-array - shape = [N_rbin - 1, N_phi]
            Incident luminosity seen at each point on disc at time t.
            IF float then assume constant irradiation across disc
            Units : W

        """
        for i in range(len(self.logr_ad_bins) - 1):
            dr_bin = 10**self.logr_ad_bins[i+1] - 10**self.logr_ad_bins[i]
            rmid = 10**(self.logr_ad_bins[i] + self.dlog_r/2)
            
            if np.ndim(Lx_t) == 0:
                Lx_r = Lx_t
            else:
                Lx_r = Lx_t[:, i]
                
            Lnu_r = self.AD_annuli(rmid, dr_bin, Lx_r)
            
            if i == 0:
                Lnu_all = Lnu_r
            else:
                Lnu_all = np.column_stack((Lnu_all, Lnu_r))
        
        if np.shape(Lnu_all) != np.shape(self.Egrid):
            Lnu_tot = np.sum(Lnu_all, axis=-1)
        else:
            Lnu_tot = Lnu_all
        
        return Lnu_tot *self.cosinc/0.5
    
    
    def DD_annuli(self, r, dr, Lx_t):
        """
        Calculates contribution from annulus in dark disc region

        Parameters
        ----------
        r : float
            Annulus midpoint - units : Rg.
        dr : float
            Annulus width - units : Rg.
        Lx_t : 1D-array - shape = shape(self.phis)
            X-ray luminosity seen at each point around the annulus at time t.

        """
        T4ann = self.calc_Trep(r, Lx_t) #Only contribution from re-processed!
        Tann_mean = np.mean(T4ann**(1/4))
        bb_ann = self.bb_radiance_ann(Tann_mean)

        Lnu_ann  = 4*np.pi*r*dr * self.Rg**2 * bb_ann #multiplying by dA to get actual normalisation
        return Lnu_ann
        
    
    def DD_spec_t(self, Lx_t):
        """
        Calculates total dark disc spectrum, given an incident luminosity Lx(t)

        Parameters
        ----------
        Lx_t : float OR 2D-array - shape = [N_rbin - 1, N_phi]
            Incident luminosity seen at each point on disc at time t.
            IF float then assume constant irradiation across disc
            Units : W

        """
        for i in range(len(self.logr_dd_bins) - 1):
            dr_bin = 10**self.logr_dd_bins[i+1] - 10**self.logr_dd_bins[i]
            rmid = 10**(self.logr_dd_bins[i] + self.dlog_r/2)
            
            if np.ndim(Lx_t) == 0:
                Lx_r = Lx_t
            else:
                Lx_r = Lx_t[:, i]
                
            Lnu_r = self.DD_annuli(rmid, dr_bin, Lx_r)
            
            if i == 0:
                Lnu_all = Lnu_r
            else:
                Lnu_all = np.column_stack((Lnu_all, Lnu_r))
        
        if np.shape(Lnu_all) != np.shape(self.Egrid):
            Lnu_tot = np.sum(Lnu_all, axis=-1)
        else:
            Lnu_tot = Lnu_all
        
        return Lnu_tot *self.cosinc/0.5
    
    
    """
    The coronal section
    Assuming this is generated through Compton scattering
    """
    
    def Corona_lumin(self):
        """
        Coronal luminosity - in this model simply dissipated accretion energy
        between r_d and risco

        """
        
        Lc, err = quad(lambda rc: sigma_sb*self.calc_Tnt(rc) * 4*np.pi*rc * self.Rg**2,
                       self.risco, self.r_d)
        
        return Lc

    def seed_temp(self):
        """
        Calculates seed photon temperature of photons being transported to
        the corona.
        For simplicity set T_seed = T_NT(risco + dr/2)

        """
        redge = 10**(np.log10(self.risco) + self.dlog_r/2) #avoiding 0 at inner edge
        T4_edge = self.calc_Tnt(redge) #inner disc T in K
        Tedge = T4_edge**(1/4)
        
        kT_edge = k_B * Tedge #units J
        kT_seed = (kT_edge * u.J).to(u.keV).value
        
        return kT_seed
    
    
    def Corona_spec(self):
        """
        Calculates spec from corona - assume generated through 
        Compton scattering (so will look like cut off power law)
        
        Not including any time-dependece in this part, as we assume all 
        corona has 0 time delay. SO will be explicitly given by input x-ray
        light-curve in method for handling time evolution
        
        """
        kTseed = self.seed_temp()
        Lum = self.Corona_lumin()
        if kTseed < self.kTe_c:
            ph_hot = donthcomp(self.Egrid, [self.gamma_c, self.kTe_c, kTseed, 1, 0])
            ph_hot = (ph_hot * u.W/u.keV).to(u.W/u.Hz, 
                                            equivalencies=u.spectral()).value
        
            Lnu_hot = Lum * (ph_hot/np.trapz(ph_hot, self.nu_grid))
        
        else:
            Lnu_hot = np.zeros(len(self.Egrid))
            
        return Lnu_hot
    
    
    
    
    ##########################################################################
    #### Spectral Evolution
    ##########################################################################
    
    def evolve_spec(self, lxs, ts):
        """
        Evolves the spectral model according to an input x-ray light-curve

        Parameters
        ----------
        lxs : 1D-array
            X-ray light-curve.
        ts : 1D-array
            Corresponding time stamps.
            
        """
        
        #First checking that beginning of time array is 0!
        if ts[0] != 0:
            ts = ts - ts[0]
        
        #Now checking that light-curve flux is fractional
        if np.mean(lxs) != 1:
            lxs = lxs/np.mean(lxs)
        
        #getting mean coronal spec - as this just goes up and down...
        #no time delay for this component...
        if self.return_corona == True:
            Lc_mean = self.Corona_spec()
        else:
            Lc_mean = np.zeros(len(self.nu_grid))
        
        
        #Now evolving light-curves!
        Lxs = lxs * self.Lx #array of x-ray lums
        Lin = np.array([self.Lx]) #array of Ls in play
        
        Lirr_ad = np.ndarray(np.shape(self.tau_ad))
        Lirr_dd = np.ndarray(np.shape(self.tau_dd))
        for j in range(len(ts)):

            Lin = np.append(Lin, [Lxs[j]])
            if j == 0:
                t_delay = np.array([np.inf])
            else:
                t_delay += ts[j] - ts[j-1] #Adding time step to delay array
            
            t_delay = np.append(t_delay, [0]) #Appending 0 for current emitted
            
            #Sorting irradiation arrays
            for k in range(len(t_delay)):
                Lirr_ad[self.tau_ad <= t_delay[k]] = Lin[k]
                Lirr_dd[self.tau_dd <= t_delay[k]] = Lin[k]
            
            #Evolving spectral components
            if self.return_AD == True and len(self.logr_ad_bins) > 1:
                Lad_t = self.AD_spec_t(Lirr_ad)
            else:
                Lad_t = np.zeros(len(self.nu_grid))
            
            if self.return_DD == True and len(self.logr_dd_bins) > 1:
                Ldd_t = self.DD_spec_t(Lirr_dd)
            else:
                Ldd_t = np.zeros(len(self.nu_grid))
            
            Lc_t = Lc_mean * lxs[j]
            Ltot_t = Lad_t + Ldd_t + Lc_t
            
            if j == 0:
                Lad_all = Lad_t
                Ldd_all = Ldd_t
                Lc_all = Lc_t
                Ltot_all = Ltot_t
            
            else:
                Lad_all = np.column_stack((Lad_all, Lad_t))
                Ldd_all = np.column_stack((Ldd_all, Ldd_t))
                Lc_all = np.column_stack((Lc_all, Lc_t))
                Ltot_all = np.column_stack((Ltot_all, Ltot_t))
        
        
        self.Lad_t_all = self._new_units(Lad_all)
        self.Ldd_t_all = self._new_units(Ldd_all)
        self.Lc_t_all = self._new_units(Lc_all)
        self.Ltot_t_all = self._new_units(Ltot_all)
        return self.Ltot_t_all
    
    
    def mean_spec(self):
        """
        Calculated the mean spectrum + each component
        Uses X-ray luminosity derived from model for irradiation
        """
        
        #disc component
        #including condition on r_bin to avoid resolution error
        #if you absoltely want to see tiny contributions then increase radial resolution
        if self.return_AD == True and len(self.logr_ad_bins) > 1:
            Lnu_ad = self.AD_spec_t(self.Lx)
        else:
            Lnu_ad = np.zeros(len(self.nu_grid))
        
        #warm component
        if self.return_DD == True and len(self.logr_dd_bins) > 1:
            Lnu_dd = self.DD_spec_t(self.Lx)
        else:
            Lnu_dd = np.zeros(len(self.nu_grid))
        
        if self.return_corona == True and self.r_c != self.risco:
            Lnu_c = self.Corona_spec()
        else:
            Lnu_c = np.zeros(len(self.nu_grid))
        
        Ltot = Lnu_ad + Lnu_dd + Lnu_c
        
        self.Lnu_ad = self._new_units(Lnu_ad)
        self.Lnu_dd = self._new_units(Lnu_dd)
        self.Lnu_c = self._new_units(Lnu_c)
        self.Lnu_tot = self._new_units(Ltot)
        return self.Lnu_tot
    
    
    def generate_lightcurve(self, band, band_width, lxs=None, ts=None):
        """
        Generated a light-curve for a band centered on nu, with bandwidth dnu
        
        Currently only does top-hat response/bandpass. Might be updated later.
        
        Input MUST be in whatever units you have set units to be!!!
        i.e if cgs of SI - then Hz,
            if counts - then keV
        IF you have NOT set any units - then default is currently SI, and you
        need to pass band in Hz
        
        NOTE: If band_width smaller than bin-width in model grid, then model bin width
        used instead!
        Parameters
        ----------
        band : float
            Midpoint in bandpass - units : Hz OR keV.
        band_width : float
            Bandwidth - units : Hz or keV.
        lxs : 1D-array, OPTIONAL
            X-ray light-curve - only needed if evolved spec NOT already calculated
        ts : 1D-array, OPTIONAL
            Light-curve time stamps

        """
        
        if hasattr(self, 'Ltot_t_all'):
            Ltot_all = self.Ltot_t_all
        else:
            if lxs == None:
                raise ValueError('NONE type light-curve not permitted!! \n'
                                 'Either run evolve_spec() FIRST \n'
                                 'OR pass a light-curve here!')
            else:    
                Ltot_all = self.evolve_spec(lxs, ts)
        
        
        if self.units == 'SI' or self.units == 'cgs':
            idx_mod_up = np.abs(band + band_width/2 - self.nu_grid).argmin()
            idx_mod_low = np.abs(band - band_width/2 - self.nu_grid).argmin()
            
            if idx_mod_up == idx_mod_low:
                Lcurve = Ltot_all[idx_mod_up, :] * band_width
                
            else:
                Lc_band = Ltot_all[idx_mod_low:idx_mod_up+1, :]
                Lcurve = np.trapz(Lc_band, self.nu_grid[idx_mod_low:idx_mod_up+1], axis=0)
            
        elif self.units == 'counts':
            idx_mod_up = np.abs(band + band_width/2 - self.Egrid).argmin()
            idx_mod_low = np.abs(band - band_width/2 - self.Egrid).argmin()
            
            if idx_mod_up == idx_mod_low:
                Lcurve = Ltot_all[idx_mod_up, :] * band_width
            
            else:
                Lc_band = Ltot_all[idx_mod_low:idx_mod_up+1, :]
                Lcurve = np.trapz(Lc_band, self.Egrid[idx_mod_low:idx_mod_up+1], axis=0)
        
        return Lcurve
        




##############################################################################
#### Sub-models (i.e extracting just disc, or just warm Comp etc..)
##############################################################################

class AGNdisc_var(AGNsed_var):
    """
    If you only want to see the standard accretion disc contribution from
    the AGNsed model
    
    Essentuially this class takes care of dealing with switching parameters
    within AGNsed_var so you dont have to!
    """
    
    def __init__(self,
                 M,
                 dist,
                 log_mdot,
                 astar,
                 cosi,
                 kTe_hot,
                 kTe_w,
                 gamma_h,
                 gamma_w,
                 r_h,
                 r_w,
                 log_rout,
                 hmax,
                 z):
        
        super().__init__(M, dist, log_mdot, astar, cosi, kTe_h, kTe_w, gamma_h, 
                         gamma_w, r_h, r_w, log_rout, hmax, z)
        
        self.return_hot = False
        self.return_warm = False


class AGNwarm_var(AGNsed_var):
    """
    If you only want to see the warm Compton contribution from
    the AGNsed model
    
    """
    
    def __init__(self,
                 M,
                 dist,
                 log_mdot,
                 astar, 
                 cosi,
                 kTe_h,
                 kTe_w,
                 gamma_h,
                 gamma_w,
                 r_h,
                 r_w,
                 log_rout,
                 hmax,
                 z):
        
        super().__init__(M, dist, log_mdot, astar, cosi, kTe_h, kTe_w, gamma_h, 
                         gamma_w, r_h, r_w, log_rout, hmax, z)
        
        self.return_disc = False
        self.return_hot = False
        

class AGNhot_var(AGNsed_var):
    """
    If you only wish to see the hot Compton contribution from the AGNsed
    model
    """
    
    def __init__(self,
                 M,
                 dist,
                 log_mdot,
                 astar,
                 cosi,
                 kTe_h,
                 kTe_w,
                 gamma_h,
                 gamma_w,
                 r_h,
                 r_w,
                 log_rout,
                 hmax,
                 z):
        
        super().__init__(M, dist, log_mdot, astar, cosi, kTe_h, kTe_w, gamma_h,
                         gamma_w, r_h, r_w, log_rout, hmax, z)
        
        self.return_disc = False
        self.return_warm = False
    


##############################################################################
#### Testing 
##############################################################################
if __name__ == '__main__': 

    
    
    import matplotlib.pyplot as plt
    
    M = 2e8
    dist = 200
    lmdot = -1.298
    astar = 0
    cosi = 0.3
    kTe_h = 100
    kTe_w = 0.2
    gamma_h = 2.03
    gamma_w = 2.593
    r_h = 26.7
    r_w = 400
    log_rout = -1
    hmax = 10
    z = 0
    
    np.random.seed(123)
    #myagn = AGNsed_var(M, dist, lmdot, astar, cosi, kTe_h, kTe_w, gamma_h, gamma_w,
    #            r_h, r_w, log_rout, hmax, z)
    
    myagn = AGNdark_var(M, dist, lmdot, astar, cosi, kTe_h, gamma_h, r_h, 
                        log_rout, hmax, z)
    
    print(myagn.risco)
    myagn.set_cgs()
    myagn.set_flux()
    ts_test = np.arange(0, 100, 1)
    lfracs = np.random.rand(len(ts_test)) + 0.5
    
    datdir = '/home/wljw75/Documents/phd/Fairall9_lightCurveCampaign/lightcurves/fourierAnalysis/F9_FourierReduced_Normalised_Interpolated_and_ReBinned_V1/'
    
    ts, fx = np.loadtxt(datdir + 'HX.dat', usecols=(0, 1), unpack=True)
    ts2, fu = np.loadtxt(datdir + 'W2.dat', usecols=(0, 1), unpack=True)

    #myagn.return_disc = False
    #myagn.return_hot = False
    Lev = myagn.evolve_spec(fx, ts)
    
    nus = myagn.nu_grid
    nu_u2 = (1928 * u.AA).to(u.Hz, equivalencies=u.spectral()).value
    idx15 = np.abs(nu_u2 - nus).argmin()
    
    lcurve_phy = myagn.generate_lightcurve(nu_u2, 0.1 * nu_u2)
    lcurve = lcurve_phy/np.mean(lcurve_phy)
    
    #plt.plot(ts_test, lfracs)
    plt.plot(ts, fx)
    plt.show()
    
    plt.plot(ts2, fu)
    plt.plot(ts, lcurve)
    plt.show()
    """
    
    #Testing spectrum
    Lnu_tot = myagn.mean_spec()
    Lnu_d = myagn.Lnu_d
    Lnu_w = myagn.Lnu_w
    Lnu_h = myagn.Lnu_h
    
    plt.loglog(nus, nus*Lnu_d, color='red', ls='-.')
    plt.loglog(nus, nus*Lnu_w, color='green', ls='-.')
    plt.loglog(nus, nus*Lnu_h, color='blue', ls='-.')
    plt.loglog(nus, nus*Lnu_tot, color='k')
    
    plt.axvspan(nu_u2 - (0.1*nu_u2)/2, nu_u2 + (0.1*nu_u2)/2, color='red', alpha=0.5)
    plt.ylim(1e-12, 1e-10)
    plt.xlim(1e14, 1e20)
    plt.show()
    
    #Testing response
    
    myagn.return_disc = False
    myagn.return_hot = False
    resp = myagn.response_components(nu_u2, 0.1*nu_u2)
    t_resp = myagn.t_imp
    
    plt.plot(t_resp, resp)
    plt.show()
    
    #testing radial response
    r_bins = myagn.logr_wc_bins
    drs = 10**r_bins[1:] - 10**r_bins[:-1]
    rmid = 10**r_bins[:-1] + drs/2
    
    print(rmid[20], drs[20])
    
    for n in range(len(rmid)):
        Tr, tau = myagn.radial_transfer(rmid[n], drs[n])
        
        if n == 0:
            Ttot_all = Tr 
        else:
            Ttot_all = np.column_stack((Ttot_all, Tr))
        
        plt.plot(tau, Tr)
    
    Ttot = np.trapz(Ttot_all, rmid, axis=-1)
    
    plt.plot(tau, Ttot, color='k')
    plt.show()
    """