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

#Generating mock xray light-curve
ts = np.arange(0, 200, 1)
xs_f = np.sin(ts/10) + 1

plt.plot(ts, xs_f)
plt.show()

#Initiating agn object
M = 2e8
mdot = 10**(-1.4)
a_star = 0
inc = 25
z = 0

mods = ['AD']
mod_rs = [20, -1]

tst_agn = AGN(M, mdot, a_star, inc, z, mods, mod_rs)
irf_comps = tst_agn.IRFcomponents(1e15, 10)
irf_dsc = irf_comps['AD']
irf_ts = tst_agn.t_imp
print(irf_dsc)
plt.plot(irf_ts, irf_dsc)
plt.show()

fs_cnv = np.convolve(irf_dsc, xs_f)

plt.plot(ts, fs_cnv)
plt.show()