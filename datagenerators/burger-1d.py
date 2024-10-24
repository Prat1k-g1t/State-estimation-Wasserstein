#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 27 10:57:46 2023

@author: prai
"""

# SPDX-FileCopyrightText: The sinkhorn-rom authors, see file AUTHORS
#
# SPDX-License-Identifier: GPL-3.0-or-later

import torch
import time
import uuid
import argparse
import numpy as np
from scipy.stats import random_correlation
from copy import deepcopy
from .parameterDomain import ParameterDomain
from .sampling import *
from ..src.utils import check_create_dir
from ..src.config import data_dir, dtype

import numpy as np
import matplotlib.pyplot as plt

def paramRange_1d(tstep):
    #TODO: Check xmin, xmax, ymin, ymax for 1D problem based on article
    # print('Flag!: caliberate paramRange()')
    tmin = 0.; tmax = 7.
    xmin = 0.; xmax = 0. #Need to caliberate
    # ymin = 0.5; ymax = 4. #y parameter, refer to Ehrlacher et al. 2020
    ymin = 2.0; ymax = 2.0 #y parameter, For transport of solution
    tmin = 0.+0.5*tstep; tmax = 0.+0.5*tstep #time parameter, For transport of solution
    # numin= 0.; numax = 0. #Inviscid Burgers
    # numin= 5.e-5; numax = 0.1 #Viscous Burgers
    # wmin = 1; wmax = 2
    return [(xmin, xmax), (ymin, ymax), (tmin,tmax)]

def explicit_soln(params,x,nx):
    """
    Explicit formula for u(t,x,y)
    """
    y = params[1]; t = params[2]
    # m = 1.
    u = torch.zeros(nx, dtype=torch.float64)
    
    yt = y*t; y_inv = 1./y
    
    #Debug
    #print(x)
    #print('sq_root', 0.<= np.sqrt(2.*t))
    
    for i,xi in enumerate(x): 

        if (t==0):
            m1 = (xi>=0.) and (xi<y_inv)
            u[i]  = y*m1

        elif (t>0) and (t <= 2./y**2):
            
            m1 = (xi>=0.) and (xi<yt)
            m2 = (xi>= yt) and (xi<(y_inv + 0.5*yt))
            u[i] = xi*m1/t + y*m2
        
        else:
            
            m1 = (xi>=0.) and (xi<=np.sqrt(2.*t))
            u[i] = xi*m1/t

    return u

def visualization(x,U):
    fig = plt.figure()
    plt.plot(x.numpy(),U.numpy().T)
    plt.show()
    #plt.pause(0.1)
    plt.close()

if __name__ == "__main__":
    """
    Generates burger-1d equation.
    """

    # Parse
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', type=int, default=10, help='number of parameters')
    args = parser.parse_args()

    # Data are always generated on cpu
    # The device is then changed during the import
    device = torch.device('cpu')
    
    xmin = -1; xmax = 4; nx = 500
    dx    = np.abs((xmax - xmin)/nx)
    #DEBUG:
    # print(dx,type(dx))

    # Spatial coordinates
    # x = torch.linspace(start=xmin, end=xmax, steps=nx, dtype=dtype, device=device)
    x = (torch.arange(nx) + 0.5)*dx + xmin

    # Parameters
    for tstep in range(11):
    # tstep = 0
        paramRange = paramRange_1d(tstep)
        paramDomain = ParameterDomain(paramRange)
        nparam = args.n
        samplingStrategy = SamplingRandom(paramDomain, nparam)
    
        # for i in range(3):
        for i in range(1):  #For transport of solution
            params = torch.tensor([p for p in samplingStrategy], dtype=dtype, device=device)
    
            # Snapshots
            print('Beginning computation of snapshots.',i)
            tic = time.time()
            fields = []
            for p in params:
                u = explicit_soln(p,x,nx)
                fields.append( u )
                visualization(x, u)
                # for j in range(explicit_soln(p,x,nx).shape[0]):  u +=explicit_soln(p,x,nx)[j]*dx
                # print('The sum of the discrete measure:',u)
            toc = time.time()
            print('End computation of snapshots. Took '+str(toc-tic)+' sec.')
    
            # Save
            # rootdir = check_create_dir(data_dir+'Burger1d/'+str(nparam)+'/'+str(i)+'/')
            rootdir = check_create_dir(data_dir+'Burger1d_transport/t{}/'.format(tstep)+str(nparam)+'/'+str(i)+'/')  #For transport of solution
            torch.save(params, rootdir+'params')
            # torch.save(x, rootdir+'points')
            torch.save(x.unsqueeze(1), rootdir+'points')
            torch.save(fields, rootdir+'fields')
            with open(rootdir+"uuid.txt", "w") as uuid_file: # save unique identifier
                uuid_file.write(uuid.uuid4().hex)

