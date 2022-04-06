#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  5 13:21:55 2022

@author: wljw75
"""

"""
Test the SED generetaded by agnvar code, and compare it to agnsed in xspec,
as these should be roughly the same
"""

import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u

import sys
sys.path.append('/home/wljw75/Documents/phd/AGNvar/src')

from agnvar import AGN


#Accretion params
r_h = 25
r_w = 100
r_out = -1 #set to r_sg
M = 2e8
mdot = 10**(-1.4)
a_star = 0
inc = np.rad2deg(np.arccos(0.9))
gamma_w = 2.5
kT_w = 0.2
gamma_h = 1.7
kT_h = 100
mods = ['HC', 'WC', 'AD']
mod_rs = [r_h, r_w, r_out]


#Calculating my model
agn_mod = AGN(M, mdot, a_star, inc, mods, mod_rs, gamma_wc=gamma_w, kT_wc=kT_w,
              gamma_hc=gamma_h, kT_hc=kT_h, skip_checks=True)

spec_comps = agn_mod.SpecComponents()
nu_tot, tot_spec = agn_mod.FullSpec()

hc_spec = spec_comps['HC']
wc_spec = spec_comps['WC']
ad_spec = spec_comps['AD']

hc_nu, hc_L = hc_spec[:, 0], hc_spec[:, 1]
wc_nu, wc_L = wc_spec[:, 0], wc_spec[:, 1]
ad_nu, ad_L = ad_spec[:, 0], ad_spec[:, 1]

hc_L = (hc_L * u.W).to(u.erg/u.s).value
wc_L = (wc_L * u.W).to(u.erg/u.s).value
ad_L = (ad_L * u.W).to(u.erg/u.s).value
#tot_spec = (tot_spec * u.W).to(u.erg/u.s).value
tot_spec = hc_L + wc_L + ad_L


#Importing XSPEC model
xs_nu, xs_fs = np.loadtxt('agnsed_xspecTest.qdp', skiprows=3, usecols=(0, 2),
                          unpack=True)
xs_n_nu, xs_n_fs = np.loadtxt('agnsed_xspecTest_noReprocessing.qdp',
                              skiprows=3, usecols=(0, 2), unpack=True)

D = 200 * u.Mpc
D = D.to(u.cm).value

xs_L = xs_fs * 4 * np.pi * D**2
xs_n_L = xs_n_fs * 4 * np.pi * D**2



#Creating plot to compare
fig = plt.figure(figsize=(10, 6))

plt.loglog(hc_nu, hc_nu*hc_L, color='blue', ls='-.', label='Hot Comp.')
plt.loglog(wc_nu, wc_nu*wc_L, color='green', ls='-.', label='Warm Comp.')
plt.loglog(ad_nu, ad_nu*ad_L, color='red', ls='-.', label='Disc')
plt.loglog(nu_tot, nu_tot*tot_spec, color='k', label='Total')
plt.loglog(xs_nu, xs_L, color='gray', ls='dotted', label='XSPEC')
plt.loglog(xs_n_nu, xs_n_L, color='black', ls='dotted', label='XSPEC, no Rep.')


plt.ylabel(r'$\nu F_{\nu}$  (erg/s)')
plt.xlabel(r'Frequency, $\nu$   (Hz)')
plt.ylim(1e42, 1e45)
plt.legend(frameon=False)
plt.show()

#print(agn_mod.fs_r)


T_NT = agn_mod.mod_dict['AD'].Td
r_gr = agn_mod.mod_dict['AD'].R_grid/agn_mod.mod_dict['AD'].Rg

plt.plot(r_gr[0, :], T_NT[0, :])
plt.show()
