#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 17:13:58 2023

@author: prai

"""

import torch
import time
import uuid
import argparse
from scipy.stats import random_correlation
from copy import deepcopy
from .parameterDomain import ParameterDomain
from .sampling import *
from ..src.utils import check_create_dir
from ..src.config import data_dir, dtype

import numpy as np
import matplotlib.pyplot as plt
    
def theta_k(a,b,v):
    k = 1.
    if v == 0.:
        return k(a-b)
    else:
        num = a*np.exp(v/(2*k))-b*np.exp(-v/(2*k))
        den = np.exp(v/(2*k))-np.exp(-v/(2*k))
        return num/den
    
def q_convolve(rho_new,rho_old,grid,xk,xl,h):
    
    sigma = 2.
    W = lambda x: -sigma*np.cos(2*math.pi*(grid-x))
    
    q = 0.
    for i in range(rho_new.shape[0]):
        q += h*0.5*(rho_new[i]+rho_old[i])*(W(xk)-W(xl))/h
        

def solver_scharfetter_gummel():
    
    N = 100
    T = 10.; dt = 0.1 
    xmin = 0.0; xmax = 1.0; h = (xmax-xmin)/N
    lmb = dt/h; tau = 1.
    rho_old = np.zeros(N)
    rho_new = np.zeros(N)
    
    for t in range(T-dt):
        for k in range(1:N-1):
            
    

if __name__ == "__main__":
    """Generates solutions to aggregation-diffusion using Scharfetter-Gummel scheme.
    """

    # Parse
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', type=int, default=10, help='number of parameters')
    args = parser.parse_args()

    # Data are always generated on cpu
    # The device is then changed during the import
    device = torch.device('cpu')
    
    if args.n == 100:
        xmin = 0; xmax = 1; nx = 1000
    else:
        xmin = 0; xmax = 1; nx = 100
    dx    = (xmax - xmin)/nx

    x = (torch.arange(nx) + 0.5)*dx + xmin

    # Parameters
    paramRange = paramRange()
    paramDomain = ParameterDomain(paramRange)
    nparam = args.n
    samplingStrategy = SamplingRandom(paramDomain, nparam)

    for i in range(3):    
        params = torch.tensor([p for p in samplingStrategy], dtype=dtype, device=device)

        # Snapshots
        print('Beginning computation of snapshots.')
        tic = time.time()
        fields = []
        for p in params:
            U = solver_burger_2d(p,grid, nmail, dh, X)
            fields.append( U )
        toc = time.time()
        print('End computation of snapshots. Took '+str(toc-tic)+' sec.')

        # Save
        rootdir = check_create_dir(data_dir+'Scharfetter-Gummel2d/'+str(nparam)+'/'+str(i)+'/')
        torch.save(params, rootdir+'params')
        torch.save(X, rootdir+'points')
        torch.save(fields, rootdir+'fields')
        with open(rootdir+"uuid.txt", "w") as uuid_file: # save unique identifier
            uuid_file.write(uuid.uuid4().hex)

