#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 10 14:18:23 2023

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
import random
from scipy.stats import random_correlation
from copy import deepcopy
from .parameterDomain import ParameterDomain
from .sampling import *
from ..src.utils import check_create_dir
from ..src.config import data_dir, dtype, results_dir
from ..src.visualization import plot_fields

import matplotlib.pyplot as plt

def paramRange(tstep):
    xmin = -3; xmax = 5.
    tmin = 0*1.e-3; tmax = 14*1.e-3
    # k2min = 16;   k2max = 18
    k2min = 16;   k2max = 16       #For transport of solution
    tmin = 0.+0.0015*tstep; tmax = 0.+0.0015*tstep #For transport of solution t = 0 to 2.5*1.e-3, delta t = 0.00025
	# return [(tmin, tmax), (c1min, c1max), (k2min, k2max)]
    return [(xmin, xmax), (k2min, k2max), (tmin, tmax)]

def A(x,c,k,t):
	A = np.zeros((2,2))
	for m in [0, 1]:
		for n in [0, 1]:
			cm = c[m]
			cn = c[n]
			km = k[m]
			kn = k[n]
			A[m,n] = (cm * cn / (km + kn)) * np.exp((km+kn)*x - (km**3 + kn**3)*t)
	return A

def dA(x,c,k,t):
	A = np.zeros((2,2))
	for m in [0, 1]:
		for n in [0, 1]:
			cm = c[m]
			cn = c[n]
			km = k[m]
			kn = k[n]
			A[m,n] = cm * cn * np.exp((km+kn)*x - (km**3 + kn**3)*t)
	return A

def d2A(x,c,k,t):
	A = np.zeros((2,2))
	for m in [0, 1]:
		for n in [0, 1]:
			cm = c[m]
			cn = c[n]
			km = k[m]
			kn = k[n]
			A[m,n] = cm * cn * (km+kn) * np.exp((km+kn)*x - (km**3 + kn**3)*t)
	return A

def explicit_soln(params,x,nx):
    ""
    "Explicit formula of u(t,x,k,c)"
    ""
    # c = np.array([2., 1.5], copy=True)
    # k = np.array([30.-params[1], params[1]], copy=True)
    
    c = np.array([2., 1.5])
    k = np.array([30.-params[1], params[1]])
    
    u = np.zeros(nx)
    
    for i,xi in enumerate(x):
        u[i] = evaluate(xi,c,k,params[2]) 

    return u

def evaluate(xi,c,k,t):
    mA   = A(xi,c,k,t)
    mdA  = dA(xi,c,k,t)
    md2A = d2A(xi,c,k,t)
    
    f   = (1.+mA[0,0]) * (1.+mA[1,1]) - mA[0,1]**2
    
    df  = mdA[0,0]*(1.+mA[1,1]) + (1.+mA[0,0])*mdA[1,1] - 2.*mA[0,1]*mdA[0,1]
    
    d2f = md2A[0,0]*(1.+mA[1,1]) + 2.*mdA[0,0]*mdA[1,1] + (1.+mA[0,0])*md2A[1,1] - 2.*mdA[0,1]**2 - 2.*mA[0,1]*md2A[0,1]
    
    return 2.*(d2f/f - df**2/f**2)

def visualization(x,U):
    fig = plt.figure()
    plt.plot(x.numpy(),U.numpy().T)
    plt.show()
    #plt.pause(0.1)
    plt.close()

if __name__ == "__main__":
    """
    Generates KDV-1d solutions.
    """

    # Parse
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', type=int, default=10, help='number of parameters')
    args = parser.parse_args()

    # Data are always generated on cpu
    # The device is then changed during the import
    device = torch.device('cpu')
    
    # if args.n == 100:
    #     xmin = -3; xmax = 5; nx = 100
    # else:
    #     # xmin = -0.5; xmax = 2; nx = 1000
    xmin = -1; xmax = 5; nx = 500
    dx    = (xmax - xmin)/nx
    #DEBUG:
    # print(dx,type(dx))

    # Spatial coordinates
    # x = torch.linspace(start=xmin, end=xmax, steps=nx, dtype=dtype, device=device)
    x = (torch.arange(nx) + 0.5)*dx + xmin

    # Parameters
    # for tstep in range(11):
    tstep = 7
    paramRange = paramRange(tstep)
    paramDomain = ParameterDomain(paramRange)
    nparam = args.n
    samplingStrategy = SamplingRandom(paramDomain, nparam)

    # for i in range(3):
    for i in range(1): #For transport of solution
        params = torch.tensor([p for p in samplingStrategy], dtype=dtype, device=device)
        # r      = random.randint(0, params.shape[0])
        # print('Shape of params',params.shape)

        # Snapshots
        print('Beginning computation of snapshots.',i)
        tic = time.time()
        fields = []
        for j,p in enumerate(params):
            u = torch.from_numpy(explicit_soln(p,x,nx))
            # print(u)
            fields.append( u )
            
            # if j == r:
            #     print('Shape of explicit_soln', explicit_soln(p,x,nx).shape)
            #     print('Shape of x', x.shape)

                # Plotting KDV solutions
            visualization(x, u)
        toc = time.time()
        print('End computation of snapshots. Took '+str(toc-tic)+' sec.')

        # Save
        # rootdir = check_create_dir(data_dir+'KdV1d/'+str(nparam)+'/'+str(i)+'/')
        rootdir = check_create_dir(data_dir+'KdV1d_transport/t{}/'.format(tstep)+str(nparam)+'/'+str(i)+'/') #For transport of solution
        torch.save(params, rootdir+'params')
        # torch.save(x, rootdir+'points')
        torch.save(x.unsqueeze(1), rootdir+'points')
        torch.save(fields, rootdir+'fields')
        with open(rootdir+"uuid.txt", "w") as uuid_file: # save unique identifier
            uuid_file.write(uuid.uuid4().hex)
            
    # del paramDomain, samplingStrategy
    
    
