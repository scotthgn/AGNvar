#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  7 17:46:09 2022

@author: wljw75
"""

"""
Module for excecuting THCOMP (Zdziarski et al. 2020) on some input spectrum.
Takes an input spectrum as two np arrays containing fluxes/luminosities and
energies. These are then loaded into pyXSPECs model manager and convolved
with THCOMP to give the output Comptonised spectrum
"""

import numpy as np
import xspec
import astropy.units as u

xspec.Xset.chatter = 0 #Stops xspec printing model info everytime it's called


def do_THCOMP(flxs, Es, flx_unit, gamma, kTe, z):
    """
    Takes input spectrum and turns it into an XSPEC model, then convolves
    it with THCOMP to porduce the comptonised spectrum
    NOTE: xspec fluxes outside of input range Es will be set to 0 
    The normalisation is chosen such that the total photon number is
    conserved.

    Parameters
    ----------
    flxs : 1D-array
        Input spectrum fluxes.
    Es : 1D-array
        Corresponding energies - units : keV.
    flx_unit : str
        Flux units (needs to be in format astropy can understand) - 
        preferably W/Hz.
    gamma : float
        Photon index of resulting Comptonised spectrum
    kTe : float
        Electron temperature (high energy roll over) - units : keV
    z : float
        Redshift

    Returns
    -------
    Es_new : 1D-array
        Output energies for new spec - units : keV
    fs_new : 1D-array
        Output fluxes for new spec - units : same as input

    """
    #Converting input fluxes to keV/s/keV - for easy conversion to counts
    flx_kev = (flxs * u.Unit(flx_unit)).to(
        u.keV/u.s/u.keV, equivalencies=u.spectral()).value
    
    flx_counts = flx_kev/Es
    tot_photon = np.trapz(flx_counts, Es) #Total photons - to be conserved
    
    #Defining the model to be loaded into XSPEC
    def modSpec(es, param, flx):
        norm = param[0]
        
        #Assume es describes left bin edges and flx evaluate at left edge
        Els = es[:-1]
        Ers = es[1:]
        
        
        #Now re-binning onto XSPEC grid
        for j in range(len(es) - 1):
            
            if Els[j] < np.amin(Es) or Els[j] > np.amax(Es):
                flx[j] = 0
            
            else:
                dE = Ers[j] - Els[j]
                idx_min = abs(Els[j] - Es).argmin()
                idx_max = abs(Ers[j] - Es).argmin()
                
                flx_in = flx_counts[idx_min:idx_max+1]

                flx[j] = (np.sum(flx_in) * dE * norm)/(
                    4*np.pi*(200e6 * 3e18)**2) #The final re-binned output
                #The division is to get in units of flux - as otherwise thcomp dies

    
    
    #Loading model into XSPEC
    parinfo = ('norm "" 1 0 0 1e20 1e20 0.01',)   
    xspec.AllModels.addPyMod(modSpec, parinfo, 'add')
    
    #Calling module
    params = (gamma, 0.5, 1, z, 1)
    mod = xspec.Model('thcomp*modSpec', setPars=params)
    mod.setPars({2:str(kTe) + ' 0.001 0.01 0.01 100 100'})
    xspec.AllData.dummyrsp(lowE=1e-4, highE=1e4, nBins=len(Es), scaleType='log')
    
    #Extracting model results
    xspec.Plot.device = '/null' #Ensuring xpec doesnt try to plot anythin
    xspec.Plot('model') #Creating the plot arrays for E and flx
    Es_new = np.array(xspec.Plot.x()) 
    flxs_new = np.array(xspec.Plot.model())
    
    #re-normalising, ie. ensuring photon count conserved, and changing units
    normC = tot_photon/np.trapz(flxs_new, Es_new)
    flx_out = ((flxs_new * normC * Es_new)*u.keV/u.s/u.keV).to(
        u.Unit(flx_unit), equivalencies=u.spectral()).value

    return Es_new, flx_out
            
    
    
   
    
    
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import DiscMods as md #just for testing
    
    r_d = 26.7369
    r_out = 400
    r_isco = 6
    hx = 10
    inc = 25.8
    mdot = 10**(-1.2989)
    eta = 0.1
    M = 2e8
    
    sp = md.DarkDisc(r_d, r_out, r_isco, hx, inc, mdot, eta, M, 0.5, 1e-4, 0.1)
    nus = sp.nu_grid
    fmod = sp.AD_spec(r_d)
    
    newNorm = 1/(4*np.pi * ((200*u.Mpc).to(u.cm))**2)
    
    fmod_erg = (fmod*u.W).to(u.erg/u.s)
    
    plt.loglog(nus, nus*fmod_erg*newNorm)
    plt.ylim(9e-14, 8e-11)
    plt.xlim(1e13, 1e18)
    
    Es = (nus*u.Hz).to(u.keV, equivalencies=u.spectral()).value
    
    Enew, f_new = do_THCOMP(fmod, Es, 'W/Hz', 2.59, 0.2, 0.045)
    
    fnew_erg = (f_new * u.W/u.Hz).to(
        u.erg/u.s/u.Hz, equivalencies=u.spectral()).value
    
    nu_new = (Enew * u.keV).to(u.Hz, equivalencies=u.spectral()).value
    
    
    plt.loglog(nu_new, fnew_erg*nu_new*newNorm)#, ls='dotted')
    
    

    
    plt.show()
    
   
