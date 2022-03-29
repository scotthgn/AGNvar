#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 29 14:43:48 2022

@author: wljw75
"""

"""
Tests the pyNTHCOMP module taken from arnauqb's qsosed repository
Basically should simply return a spectrum for some input energy array

Compares to a output file from NTHCOMP in XSPEC, for the same input params
"""

import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.append('/home/wljw75/Documents/phd/AGNvar/model_bin')

import pyNTHCOMP as nt

Es = np.geomspace(0.1, 1e3, 1000)
params = [1.7, 100, 0.2, 1, 0]

ph = nt.donthcomp(Es, params)

#Normalising s.t 1 at 1keV - just like nthcomp
idx1 = np.abs(Es - 1).argmin()
normC = 1/ph[idx1]
ph = normC * ph


#importing xspec nthcomp
E_xs, ph_xs = np.loadtxt('nthcomp_testres.qdp', skiprows=3, usecols=(0, 2),
                         unpack = True)

plt.loglog(Es, Es * ph)
plt.loglog(E_xs, ph_xs)
plt.show()