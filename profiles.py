#!/usr/bin/python
# Author:       prepared by K Cassou
# Date:         2021-03-04
# Purpose:      set of function for plasma target parameter definition 
# Source:       Python 3 (python2 on IRENE)
#####################################################################

### loading module
from __future__ import (division, print_function, absolute_import,unicode_literals)
import os,sys
import numpy as np
import scipy.constants as sc
import matplotlib.pyplot as plt


## plasma profile

def plasmaProfile(ne1, L1, r, Lx, laser_fwhm, lambda_0, print_flag= False):
    """
    return the longitudinal plasma profile of a chair-like target 
    ne1 = first plateau electron density    [m^-3]
    L1 = length of the first plateau        [m]
    r = ne2/ne1                             [m^-3]
    LX grid length in smilei unit
    Laser fwhm length in smilei unit
    return numpy array (x,ne)  in m and     m^-3
    """
    # conversion mm2m
    mm2m = 1e-3
    # polygonal characteristics point 
    x0 = 0.0
    xupramp1 = (Lx+1.2*laser_fwhm)*lambda_0/(2*pi) # starting point
    lupramp1 = 1.e-3  # first up ramp  
    xupramp2 = xupramp1+lupramp1
    lupramp2 = 0.7e-3  #second upramp length of the input diameter l2,d2
    xupramp3 = xupramp2+lupramp2
    lupramp3 = 0.3e-3   #third upramp
    # plateau region #  
    xplateau1 = xupramp3 + lupramp3
    lplateau1 = 0.85*L1                 #region 1 plateau length 
    # downramp 1 fix by the aperture 1->2 slope variation as function of ne2/ne1 neglated
    xbegindownramp1 = xplateau1+lplateau1 
    ldownramp1 = 0.35e-3
    ldownramp2 = 1.0e-3
    # plateau region 2 with correction due to non null flow 1->2
 
    ne_up1 = 0.5*ne1
    ne_up2 = 0.75*ne1

    l1 = [ 0.52644434, -0.60966651, -0.1897902,   0.58150618]
    x1 = poly1d(l1)*mm2m + xbegindownramp1 
    l2 = [ 0.84934688, -0.82402032, -0.30158281,  0.75062812]
    x2 = poly1d(l2)*mm2m + xbegindownramp1 
    l3 = [ 0.26187251, -0.21280877,  0.15913015,  0.62090305]
    x3 = poly1d(l3)*mm2m + xbegindownramp1 
    x4 = 0.72*mm2m + xbegindownramp1 
    x5 = 1.49*mm2m + xbegindownramp1 
    xend = x5 + ldownramp2
    
    k1 = [-2.94467180e+24,  9.97007699e+23,  1.26329259e+24,  5.71857555e+22]
    y1 = r*ne1 + poly1d(k1)
    k2 = [ 2.97081767e+23, -8.05622651e+23, -1.45440946e+23,  8.26990915e+23]
    y2 = r*ne1 + poly1d(k2)
    k3 = [ 9.04505418e+23, -1.17675813e+24, -4.56746087e+23,  9.29499398e+23]
    y3 = r*ne1 + poly1d(k3)
    k4 = [ 9.42236655e+23, -1.05335498e+24, -4.59495749e+23,  7.85307038e+23]
    y4 = r*ne1 + poly1d(k4)
    k5 = [-1.00304010e+24,  6.45601489e+22,  2.95582151e+23,  1.21448900e+23]
    y5 = r*ne1 + poly1d(k5)
    
    xr = array([x0, xupramp1, xupramp2, xupramp3, xplateau1, xbegindownramp1,
                x1(r), x2(r), x3(r), x4, x5, xend])
    
    ner = array([0,0,ne_up1,ne_up2,ne1,ne1,
                    y1(r), y2(r), y3(r), y4(r), y5(r), 0])
    
    if print_flag == True:
        print("###########################################################\n",
        xr,
        ner,
        "\n ###########################################################")

    return xr, ner 

## dopant profile with leak correction 

def dopantProfile(C_N2,ne1,r,xr,ner,print_flag= False):
    """ return the longitudinale profile of dopant taking into account the
    a rough correction for the leak depending on the ratio of r = ne2/ne1
    return a numpy array (x,nN2) [m,cm^-3]
    """
    # correction of the density compared to null flow between region 1 and 
    # region 2
    correction1 = [ 0.73285714, -1.07571428,  0.41714286]
    correction2 = [-0.03891875,  0.26215569, -0.07071524]

    pcorrection1 = poly1d(correction1)
    pcorrection2 = poly1d(correction2)
    # correction factor for zone 1 density
    xN2 = xr 
    nN2 = ner*C_N2

    nN2[6:] = ner[6:]*C_N2* pcorrection1(r)

    # diffusion correction 
    nN2[-2] = ner[-2]*C_N2* pcorrection1(r)*pcorrection2(r)
    
    if print_flag == True:
        print("###########################################################\n",
        xN2,
        nN2,
        "\n ###########################################################")

    return xN2,nN2
