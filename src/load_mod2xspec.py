#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  5 12:10:14 2022

@author: wljw75
"""

"""
Incorporates methods for loading the model into pyXSPEC
This is mostly if you want to fit the time-averaged spectum using without
figuring out the relevant XSPEC models to use...
"""

import xspec
import numpy as np
import agnvar


##############################################################################
#---- Functions for checking energy grid and casting between grids
##############################################################################

def _check_ear(ear):
    """
    Checks if input energy grid extends beyond the default
    If yes, then passes new limits

    Parameters
    ----------
    ear : 1D-array
        XSPEC energy grid.

    """
    
    if min(ear) < 1e-4:
        new_emin = min(ear)
    else:
        new_emin = 1e-4
    
    if max(ear) > 1e4:
        new_emax = max(ear)
    else:
        new_emax = 1e4
    
    return new_emin, new_emax


def _interp_grid(Ei, Emod, flxs):
    """
    Interpolated from model grid onto XSPEC energy grid

    Parameters
    ----------
    Ei : float
        Mid-point in XSPEC energy bin.
    Emods : 1D-array
        Energy grid used in model.
    flxs : 1D-array
        Model fluxes
    """
    
    idx_1 = np.abs(Ei - Emod).argmin()
    
    if Ei - Emod[idx_1] > 0:
        if Emod[idx_1] != Emod[-1]: #ensuring we dont fall off array
            E1 = Emod[idx_1]
            E2 = Emod[idx_1 + 1]
            f1 = flxs[idx_1]
            f2 = flxs[idx_1 + 1]
        
        else:
            E1 = Emod[idx_1 - 1]
            E2 = Emod[idx_1]
            f1 = flxs[idx_1 -1]
            f2 = flxs[idx_1]
        
        df_dE = (f2 - f1)/(E2 - E1)
        fi = df_dE * (Ei - E1) + f1
    
    elif Ei - Emod[idx_1] < 0:
        if Emod[idx_1] != Emod[0]:
            E1 = Emod[idx_1 - 1]
            E2 = Emod[idx_1]
            f1 = flxs[idx_1 -1]
            f2 = flxs[idx_1]
        
        else:
            E1 = Emod[idx_1]
            E2 = Emod[idx_1 + 1]
            f1 = flxs[idx_1]
            f2 = flxs[idx_1 + 1]
            
        df_dE = (f2 - f1)/(E2 - E1)
        fi = df_dE * (Ei - E1) + f1
    
    else:
        fi = flxs[idx_1]
    
    return fi



##############################################################################
#---- Functions for loading to XSPEC
##############################################################################

def load_pyAGNSED():
    """
    Loads this version of AGNSED into XSPEC
    (although not much point as you might as well just use AGNSED...)

    """
    
    def pyagnsed(ear, params, flx):
        M = params[0] #Msol
        D = params[1] #Mpc
        l_mdot = params[2] #L/Ledd
        astar = params[3]
        cosi = params[4]
        kTe_h = params[5] #keV
        kTe_w = params[6] #keV
        gamma_h = params[7]
        gamma_w = params[8]
        r_h = params[9] #Rg
        r_w = params[10] #Rg
        log_rout = params[11] #Rg
        hmax = params[12] #Rg
        z = params[13]
        
        
        Els = np.array(ear[:-1])
        Ers = np.array(ear[1:])
        dEs = Ers - Els
        Emid = Els + 0.5*dEs #evaluate model at center of bin
        
        
        magn = agnvar.AGNsed_var(M, D, l_mdot, astar, cosi, kTe_h, kTe_w, 
                                 gamma_h, gamma_w, r_h, r_w, log_rout, hmax, z)
        
        emin, emax = _check_ear(ear)
        magn.new_ear(np.geomspace(emin, emax, 1000))
        
        magn.set_counts()
        magn.set_flux()
        
        flxs_mod = magn.mean_spec()
        print(flxs_mod)
        for i in range(len(Emid)):
            flx[i] = _interp_grid(Emid[i], magn.Egrid, flxs_mod)
        
        
    parinfo_pyagnsed = ('M Msol 1e7 10 10 1e10 1e10 -1',
                         'dist Mpc 1 1e-3 1e-3 1e3 1e3 -1',
                         'log_mdot "" -1 -10 -10 2 2 -1',
                         'astar "" 0 -0.998 -0.998 0.998 0.998 0.001',
                         'cosi "" 0.5 0.09 0.09 0.998 0.998 0.001',
                         'kTe_h keV 100 10 10 300 300 -1',
                         'kTe_w keV 0.2 0.01 0.01 1 1 0.001',
                         'gamma_h "" 1.7 1.3 1.3 3 3 0.01',
                         'gamma_w "" 2.7 2 2 5 10 0.01',
                         'r_h Rg 10 6 6 500 500 0.01',
                         'r_w Rg 20 6 6 500 500 0.01',
                         'logrout Rg -1 -1 -1 7 7 -1',
                         'hmax Rg 10 6 6 10 10 -1',
                         'z "" 0 0 0 10 10 -1')
    
    xspec.AllModels.addPyMod(pyagnsed, parinfo_pyagnsed, 'add')
    print('pyAGNSED loaded succesfully!')
    print('Call as: pyagnsed')
    
    
    
    
    