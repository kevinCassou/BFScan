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


def plasmaProfile(ne1, L1, ne2, Lx, laser_fwhm, print_flag= False, plot_flag = False):
    """
    return the longitudinal plasma profile of a chair-like target 
    ne1 = first plateau electron density    [cm^-3]
    L1 = length of the first plateau        [m]
    ne2 = density of the second plateau     [cm^-3]
    return numpy array (x,ne)  in m and cm^-3
    """
    # conversion mm2m
    mm2m = 1e-3
    
    xupramp1 = Lx+1.2*laser_fwhm # starting point
    lupramp1 = 1.e-3  # first up ramp  
    xupramp2 = xupramp1+lupramp1
    lupramp2 = 0.7e-3  #second upramp length of the input diameter l2,d2
    xupramp3 = xupramp2+lupramp2
    lupramp3 = 0.3e-3   #thrid upramp  
    xplateau1 = xupramp3 + lupramp3
    lplateau1 = 0.85e-3
    xbegindownramp1 = xplateau1+lplateau1
    ldownramp1 = 0.35e-3
    xplateau2 = xbegindownramp1 + ldownramp1
    lplateau2 = 0.8*ldownramp1
    xbegindownramp2 = xplateau2 + lplateau2
    ldownramp2 = ldownramp1/2.
    xbegindownramp3 = xbegindownramp2 + ldownramp2
    ldownramp3 = 0.7e-3
    xbegindownramp4 = xbegindownramp3 + ldownramp3
    ldownramp4 = 1e-3
    xend = xbegindownramp4 + ldownramp4
    ne_up1 = 0.5*ne1
    ne_up2 = 0.75*ne1
    r = ne2/ne1
    l1 = [ 0.60708368, -0.52921582, -0.29939209,  0.60611454]
    x1 = np.poly1d(l1)
    l2 = [ 0.86111787, -0.69143663, -0.41828169,  0.77436027]
    x2 = np.poly1d(l2)
    l3 = [ 0.080351  , -0.12722241,  0.18361121,  0.60995613]
    x3 = np.poly1d(l3)
    x4 = 0.72
    x5 = 1.49
    
    k1 = [-7.64947689e+18,  5.05567175e+18,  6.36416413e+18,  8.01846117e+16]
    y1 = np.poly1d(k1)
    k2 = [-5.17550459e+18,  3.73054893e+18,  4.96272554e+18,  8.22938073e+17]
    y2 = np.poly1d(k2)
    k3 = [-4.65863584e+18,  3.44017341e+18,  4.63191329e+18,  9.26329480e+17]
    y3 = np.poly1d(k3)
    k4 = [-4.74272589e+18,  3.57331599e+18,  4.68534201e+18,  7.66840736e+17]
    y4 = np.poly1d(k4)
    k5 = [-6.19788595e+18,  4.38931942e+18,  5.43334029e+18,  1.20735744e+17]
    y5 = np.poly1d(k5)
    
    xr = np.array([x0,xupramp1,xupramp2,xupramp3,xplateau1,xbegindownramp1,
                   xbegindownramp1 + x1(r)*mm2m,
                   xbegindownramp1 + x2(r)*mm2m,
                   xbegindownramp1 + x3(r)*mm2m,
                   xbegindownramp1 + x4*mm2m,
                   xbegindownramp1 + x5*mm2m,
                   xend])
    
    ner = np.array([0,0,ne_up1,ne_up2,ne1,ne1,
                    y1(r),
                    y2(r),
                    y3(r),
                    y4(r),
                    y5(r),
                    0])

    if plot_flag == True:
        fig, ax = plt.subplots()
        ax.plot(xr,ner)
        ax.set_xlabel('x[m]')
        ax.set_ylabel('ne (x)')
    
    if print_flag == True:
        print("###########################################################\n"
        ,xr,
        ,ner,
        "\n ###########################################################")

    return np.vstack((xr,ner))

# dopant profile with leak correction 

def dopantProfile(C_N2,ne1,ne2,xr,ner):
    """ return the longitudinale profile of dopant taking into account the
    a rough correction for the leak depending on the ratio of ne2/ne1
    """
    # correction of the density compared to null flow between region 1 and 
    # region 2
    r = ne2/ne1
    correction1 = [ 0.73285714, -1.07571428,  0.41714286]
    correction2 = [-0.03891875,  0.26215569, -0.07071524]

    pcorrection1 = np.poly1d(correction1)
    pcorrection2 = np.poly1d(correction2)
    # correction factor for zone 1 density
    xN2 = xr 
    nN2 = ner*C_N2

    nN2[6:] = ner[6:]*C_N2* pcorrection1(r)

    # diffusion correction 
    nN2[-2] = ner[-2]]*C_N2* pcorrection1(r)*pcorrection2(r)


    if plot_flag == True:
        fig, ax = plt.subplots()
        ax.plot(xN2,nN2)
        ax.set_xlabel('x[m]')
        ax.set_ylabel('ne (x)')
    
    if print_flag == True:
        print("###########################################################\n"
        ,xN2,
        ,nN2,
        "\n ###########################################################")

    return np.vstack((xN2,nN2))

 
