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
import xspec #so can compare to agnsed


#Accretion params
r_h = 25
r_w = 100
r_out = -1 #set to r_sg
M = 2e8
mdot = 10**(-1.4)
a_star = 0
inc = np.rad2deg(np.arccos(0.9))
z=0
gamma_w = 2.5
kT_w = 0.2
gamma_h = 1.7
kT_h = 100
mods = ['HC', 'WC', 'AD']
mod_rs = [r_h, r_w, r_out]


#Calculating my model
agn_mod = AGN(M, mdot, a_star, inc, z, mods, mod_rs, gamma_wc=gamma_w, kT_wc=kT_w,
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


#Calculating AGNSED (for comparison)
xspec.Xset.chatter = 0
xspec.AllData.dummyrsp(1e-4, 1e3, 1000)

mpars = (M, 200, -1.4, a_star, 0.9, kT_h, kT_w, gamma_h, gamma_w, r_h, r_w, -1, 10, 1, z)
xspec.Model('agnsed', setPars=mpars)

xspec.Plot.device = '/null'
              
xspec.Plot('model')
Es = np.array(xspec.Plot.x())
phs = np.array(xspec.Plot.model())

ph_f = phs*Es
xs_fs = (ph_f * u.keV/u.s/u.keV).to(u.W/u.Hz, equivalencies=u.spectral()).value
xs_nu = (Es*u.keV).to(u.Hz, equivalencies=u.spectral())

D = 200 * u.Mpc
D = D.to(u.cm).value

xs_L = xs_fs * 4 * np.pi * D**2




#Creating plot to compare
fig = plt.figure(figsize=(10, 6))

plt.loglog(hc_nu, hc_nu*hc_L, color='blue', ls='-.', label='Hot Comp.')
plt.loglog(wc_nu, wc_nu*wc_L, color='green', ls='-.', label='Warm Comp.')
plt.loglog(ad_nu, ad_nu*ad_L, color='red', ls='-.', label='Disc')
plt.loglog(nu_tot, nu_tot*tot_spec, color='k', label='Total')
plt.loglog(xs_nu, xs_nu*xs_L*1e7, color='gray', ls='dotted', label='XSPEC')
#plt.loglog(xs_n_nu, xs_n_L, color='black', ls='dotted', label='XSPEC, no Rep.')


plt.ylabel(r'$\nu F_{\nu}$  (erg/s)')
plt.xlabel(r'Frequency, $\nu$   (Hz)')
plt.ylim(1e42, 1e45)
plt.legend(frameon=False)
plt.show()

