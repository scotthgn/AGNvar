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
from pyNTHCOMP import donthcomp
from DiscMods import Disc, CompDisc


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
        
        
        #Calculating disc params
        self._calc_Ledd() #eddington luminosity
        self._calc_risco() #innermost stable curcular orbit
        self._calc_efficiency() #accretion efficiency
        self._calc_r_selfGravity() #Self gravity radius
        
        
        
if __name__ == '__main__':
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
    print(adaf_mod.r_isco)
        