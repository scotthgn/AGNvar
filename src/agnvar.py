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
from model_bin.DiscMods import Disc, CompDisc




class AGN:
    """
    Defines an AGN object for you to do 'funky' science on!
    """
    
    def __init__(self,
                 M,
                 mdot,
                 a_star,
                 inc,
                 mods,
                 mod_rs,
                 gamma_wc=2.1,
                 kT_wc=0.2,
                 skip_checks=False):
        """
        Initiates the AGN object - defines geometry

        Parameters
        ----------
        M : float
            BH mass - units : Msol.
        mdot : float
            Accretion rate - units : Medd.
        a_star : float
            Dimensionless BH spin parameter - between [-1, 1].
        inc : float
            Inclination - units : deg.
        mods : list
            List containing model strings you wish to implement
            Needs to be ordered from inner region to outer region
                e.g [WC, AD] would create a warm compton inner region
                followed by a standard accretion disc
            
            Current available models are:
                AD - Stanard accretion disc, using Novikov-Thorne emissivity
                DD - Darkened accretion disc, so defaults r_in = r_isco, 
                    however for r<r_dark the intrinsic emission is transported
                    directly to the corona. Hence below r_dark one would only
                    see contributions from the re-processed emission
                WC - Warm Comptonised region, a disc struggling to thermalise.
                    We assume an underlying disc generates the seed photons,
                    following Novikov-Thorne, however these are then Compton
                    scattered by a corona above the disc, giving rise to a 
                    Comptonised spectrum.
                    
        mod_rs : list
            List containing the radii with which to place each model component.
            Need to define the inner + outer radius of the first model, and then
            only the outer radii of the following models. It is assumed that 
            the next component begins where the previous ends. If the final 
            radius (so outermost radius) is set to -1, then the self-gravity
            radius is used. If first radius set to -1, then r_isco is used for
            r_in.
                
                e.g: if mods = [WC, AD] and mod_rs = [25, 400, -1], then 
                that would generate a warm compton region between 25 to 400 Rg,
                followed by a standard accretion disc extending from 400Rg to 
                the self-gravity radius
            
            Units : Rg
        
        gamma_wc : float, OPTIONAL, DEFAULT=2.1
            Photon index for warm compton model
        kT_wc : float, OPTIOANL, DEFAULT=0.2
            Electron temperature (high-energy roll over) for warm Compton 
            model - units : keV
        """
        
        #Read parameters
        self.M = M 
        self.mdot = mdot
        self.a = a_star
        self.inc = inc
        self.gamma_wc = gamma_wc
        self.kT_wc = kT_wc
        
        self.mods = mods
        self.mod_rs = mod_rs

        
        #checking correct number of radii to models
        self._check_NumRadii()
        
        #Initiaing models
        self.mod_dict = {}
        for i in range(len(self.mods)):
            if self.mods[i] == 'AD' or self.mods[i] == 'DD':
                self.mod_dict[self.mods[i]] = Disc(self.mod_rs[i],
                            self.mod_rs[i+1], self.a, self.inc,
                            self.mdot, self.M, model=self.mods[i],
                            skip_checks=skip_checks)
            
            elif self.mods[i] == 'WC':
                self.mod_dict[self.mods[i]] = CompDisc(self.mod_rs[i], 
                            self.mod_rs[i+1], self.a, self.inc, 
                            self.mdot, self.M, self.gamma_wc, self.kT_wc,
                            skip_checks=skip_checks)
        
        
        
        #Extracting accretion disc parameters IF AD, DD, or WC model used
        for j in self.mods:
            if j in ['AD', 'DD', 'WC']:
                d_mod = self.mod_dict[j]
                self.r_isco = d_mod.r_isco
                self.eta = d_mod.eta
                self.L_edd = d_mod.L_edd
                self.r_sg = d_mod.r_sg
                self.Mdot_edd = d_mod.Mdot_edd
                
                self.Lx = d_mod.Lx 
                
                break
        
        
        #Updating Lx for other models
        for k in range(len(self.mods)):
            if k != 0:
                self.mod_dict[self.mods[k]].change_Lx(self.Lx)
        
        
        
        
        
    def _check_NumRadii(self):
        try:
            assert len(self.mod_rs) == len(self.mods) + 1
        except:
            raise AssertionError('Number of radii given does not sufficiently'
                                 ' define the model geometry!! \n'
                                 '\n'
                                 'Require Num_radii = 1 + Num_mods')
    
    
    
    
    """
    Section for creating model spectra
    """
    
    def SpecComponents(self):
        """
        Calculates the spectrum for each model component

        Returns
        -------
        spec_comp : dict
            Dictionary containing spectra for each model component.

        """
        
        self.spec_comp = {}
        for i in self.mods:
            #WC is inititalised slightly differently to AD or DD
            if i == 'WC':
                Lx_mod = self.mod_dict[i].Lx
                Lnu_mod = self.mod_dict[i].Calc_spec(Lx_mod)
            else:
                Lnu_mod = self.mod_dict[i].Calc_spec()
                
            nu_mod = self.mod_dict[i].nu_grid
            
            self.spec_comp[i] = np.column_stack((nu_mod, Lnu_mod))
            
        return self.spec_comp
    
    
    def FullSpec(self):
        """
        Adds together all the spectral components and return total spec

        Returns
        -------
        Lnu : 1D-array
            Total output spectrum

        """
        
        #Avioding calculating spectral components twice...
        if self.spec_comp:
            pass
        else:
            self.SpecComponents()
            
        for i in range(len(self.mods)):
            Lnu_mod = self.spec_comp[self.mods[i]]
            if i == 0:
                Lnu_tot = Lnu_mod[:, 1]
                nus = Lnu_mod[:, 0]
            else:
                Lnu_tot = Lnu_tot + Lnu_mod[:, 1]
        
        return nus, Lnu_tot
    
    
    """
    Section for calculating impulse response functions
    """
    def IRFcomponents(self, nu, dnu):
        """
        Calculates the impulse response for each disc component for a band
        centered on frequency nu, with bandwidth dnu

        Parameters
        ----------
        nu : float
            Band central frequency - units : Hz.
        dnu : float
            Band width - units : Hz.

        Returns
        -------
        irf_comp : dict
            Dictionary containing impulse response functions for each model
            component

        """
        
    



if __name__ == '__main__': 
    import matplotlib.pyplot as plt
    
    M = 2e8
    mdot = 10**(-1.4)
    a_star = 0
    inc = 25
    
    mods = ['WC', 'AD']
    mod_rs = [26, 400, -1]
    
    agn_mod = AGN(M, mdot, a_star, inc, mods, mod_rs, gamma_wc=2.5, kT_wc=0.2,
                  skip_checks=True)
    
    spec_comps = agn_mod.SpecComponents()
    nus, Lnu = agn_mod.FullSpec()
    
    wc_spec = spec_comps['WC']
    ad_spec = spec_comps['AD']
    
    wc_nu, wc_L = wc_spec[:, 0], wc_spec[:, 1]
    ad_nu, ad_L = ad_spec[:, 0], ad_spec[:, 1]
    
    tot_L = wc_L + ad_L
    
    plt.loglog(wc_nu, wc_nu * wc_L, color='green', ls='-.')
    plt.loglog(ad_nu, ad_nu * ad_L, color='red', ls='-.')
    plt.loglog(ad_nu, ad_nu * tot_L, color='k')
    plt.loglog(nus, nus*Lnu)
    plt.ylim(1e35, 1e38)
    plt.show()
    
    
    
    
    
    
    
    