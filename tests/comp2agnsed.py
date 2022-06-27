#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  8 11:49:35 2022

@author: wljw75

Compares mean sed to agnsed - for testing!
"""

import numpy as np
import matplotlib.pyplot as plt
import xspec

import sys
sys.path.append('/home/wljw75/Documents/phd/AGNvar/src')

from agnvar import AGNsed_var


#test params
M = 2e8
dist = 200
log_mdot = -1.2
astar = 0.998
cosi = 0.9
kTe_h = 100
kTe_w = 0.4
gamma_h = 1.9
gamma_w = 2.8
r_h = 10
r_w = 50
log_rout = -1
hmax = 10
z = 4.5e-2


magn = AGNsed_var(M, dist, log_mdot, astar, cosi, kTe_h, kTe_w, gamma_h,
                  gamma_w, r_h, r_w, log_rout, hmax, z)

magn.set_counts()
magn.set_flux()

Lnu = magn.mean_spec()
Eobs = magn.E_obs


#xspec agnsed version
xspec.Xset.chatter = 0
xspec.AllData.dummyrsp(1e-4, 1e4)

agnpars = (M, dist, log_mdot, astar, cosi, kTe_h, kTe_w, gamma_h, gamma_w,
           r_h, r_w, log_rout, hmax, 1, z)
xspec.Model('agnsed', setPars=agnpars)

xspec.Plot('mo')
Exs = np.array(xspec.Plot.x())
ph = np.array(xspec.Plot.model())



#Plotting
plt.loglog(Eobs, Eobs**2 * Lnu, label='agnvar')
plt.loglog(Exs, Exs**2 * ph, label='agnsed')

plt.ylim(1e-4, 1e-1)
plt.legend(frameon=False)
plt.show()

