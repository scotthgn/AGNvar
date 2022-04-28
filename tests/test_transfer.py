#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 26 14:24:07 2022

@author: wljw75
"""

"""
Tests that transfer function gives same result as the 'longwinded calculation'
"""

import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u

import sys
sys.path.append('/home/wljw75/Documents/phd/AGNvar/src')

from agnvar import AGN
from model_bin.DiscMods import Disc


r = 10
h = 10
M = 2e8
c = 3e8
G = 6.67e-11 * 1.99e30
inc = 0.5

Rg = (G*M)/c**2

tau_min = (Rg/c) * (np.sqrt(r**2 + h**2) + h*np.cos(inc) - r*np.sin(inc))
tau_max = (Rg/c) * (np.sqrt(r**2 + h**2) + h*np.cos(inc) + r*np.sin(inc))

taus = np.linspace(tau_min, tau_max, 10)

def dphi(tau):
    root = 1 - ((1/(r*np.sin(inc))) * (np.sqrt(r**2 + h**2) + h*np.cos(inc) - (c*tau)/Rg))**2
    dtaus
    return -1/(np.sqrt(root))

print(dphi(taus))