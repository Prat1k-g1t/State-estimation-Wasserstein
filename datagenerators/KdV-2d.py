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

def fromIndexToCell(index, nmail):
    icell = index[1]*nmail[0]+index[0]
    return icell

def getCellIndex(icell, nmail):
    index = [icell % nmail[0], icell // nmail[0]]
    return index 

def spatial_coordinates_2d(xmin, xmax, nmail):
    Ndim  = nmail.shape[0]
    ncell = torch.prod(nmail)
    grid= [torch.linspace(xmin[idim], xmax[idim], nmail[idim]+1) for idim in range(Ndim)]
    dh    = (xmax-xmin)/nmail
    points = torch.zeros((ncell,2))
    for j in range(nmail[1]):
        for i in range(nmail[0]):
            xi = xmin[0] + (i+0.5)*dh[0]
            vj = xmin[1] + (j+0.5)*dh[1]
            icell = fromIndexToCell([i, j], nmail)
            points[icell, 0] = xi
            points[icell, 1] = vj

    return grid, dh, points


def paramRange():
    xmin = -0.5; xmax = 2.0
    tmin = 0.; tmax = 2.5*1.e-3
	# c1min = 2.; c1max = 2.
    k2min = 16;   k2max = 22
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

def explicit_soln(params,x,y,nx):
    ""
    "Explicit formula of u(t,x,y,k,c)"
    ""
    
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

def visualization(grid,dh, U):
    fig = plt.figure()
    #ax = plt.axes(projection='3d')
    x = grid[0][:-1]+0.5*dh[0]
    y = grid[1][:-1]+0.5*dh[1]
    X,Y = np.meshgrid(x.numpy(), y.numpy())
    #ax.plot_wireframe(X,Y,U.numpy().T, cmap='hsv')
    #ax.plot_surface(X,Y,U.numpy().T, cmap='hot')
    # print('X',X)
    # print('Y',Y)
    # print('U.shape',U.shape)
    
    plt.pcolor(X,Y,U.numpy().T,cmap='jet',shading='auto')
    plt.colorbar()
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
    #     xmin = -3; xmax = 5; nx = 1000
    # else:
    #     xmin = -0.5; xmax = 2; nx = 100
        
    nx   = 100
    xmin = torch.tensor([-0.5,-0.5])
    xmax = torch.tensor([2.0 ,2.0])
    nx   = torch.tensor([100 ,100])
        
    dx    = (xmax - xmin)/nx
    
    # Spatial coordinates
    grid, dh, X = spatial_coordinates_2d(xmin,xmax,nmail)

    # Parameters
    paramRange = params_range_2d(num_rect=1)
    paramDomain = ParameterDomain(paramRange)
    nparam = args.n
    samplingStrategy = SamplingRandom(paramDomain, nparam)

    for i in range(3):    
        params = torch.tensor([p for p in samplingStrategy], dtype=dtype, device=device)
        # r      = random.randint(0, params.shape[0])
        # print('Shape of params',params.shape)

        # Snapshots
        print('Beginning computation of snapshots.',i)
        tic = time.time()
        fields = []
        for j,p in enumerate(params):
            u = torch.from_numpy(explicit_soln(p,grid,nx))
            # print(u)
            fields.append( u )
            
            # if j == r:
            #     print('Shape of explicit_soln', explicit_soln(p,x,nx).shape)
            #     print('Shape of x', x.shape)

                # Plotting KDV solutions
                # visualization(x, u)
        toc = time.time()
        print('End computation of snapshots. Took '+str(toc-tic)+' sec.')

        # Save
        rootdir = check_create_dir(data_dir+'KdV2d/'+str(nparam)+'/'+str(i)+'/')
        torch.save(params, rootdir+'params')
        # torch.save(x, rootdir+'points')
        torch.save(x.unsqueeze(1), rootdir+'points')
        torch.save(fields, rootdir+'fields')
        with open(rootdir+"uuid.txt", "w") as uuid_file: # save unique identifier
            uuid_file.write(uuid.uuid4().hex)


