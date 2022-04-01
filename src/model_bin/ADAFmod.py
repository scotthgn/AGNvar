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



class ADAF:
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
                 seed_type='WC'):