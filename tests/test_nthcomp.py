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

Es = np.geomspace(1e-4, 1e4, 10000)
params = [1.7, 100, 0.2, 1, 0]

ph = nt.donthcomp(Es, params)

#Normalising s.t 1 at 1keV - just like nthcomp
idx1 = np.abs(Es - 1).argmin()
normC = 1/ph[idx1]
ph = normC * ph


#importing xspec nthcomp
E_xs, ph_xs = np.loadtxt('nthcomp_testres.qdp', skiprows=3, usecols=(0, 2),
                         unpack = True)

ph2 = nt.donthcomp(Es, [1.8, 100, 0.2, 1, 0])


fig = plt.figure(figsize=(5, 6))
grid = plt.GridSpec(3, 1, hspace=0)

ax = fig.add_subplot(grid[:2, :])
subax = fig.add_subplot(grid[2, :], sharex=ax)

ax.loglog(Es, Es * ph, label='pyNTHCOMP')
ax.loglog(E_xs, ph_xs, ls='dashed', label='XSPEC NTHCOMP')
ax.loglog(Es, Es * ph2 * normC)
ax.set_ylim(1e-3, 10)
ax.set_xlim(1e-2, 1e4)
ax.legend(frameon=False)
ax.set_ylabel(r'E F(E)   keV$^{2}$ (Photons cm$^{-2}$ s$^{-1}$ keV$^{-1}$)')
ax.tick_params(axis='x', labelsize=0, labelcolor='white', direction='inout')

frac_diff = (ph - ph_xs)/(0.5 * (ph + ph_xs))
subax.plot(Es, frac_diff)
subax.axhline(0, ls='-.', color='gray')

subax.set_ylim(-3, 3)
subax.set_ylabel('Frac. Diff.')
subax.set_xlabel('Energy (keV)')

plt.tight_layout()
plt.show()


