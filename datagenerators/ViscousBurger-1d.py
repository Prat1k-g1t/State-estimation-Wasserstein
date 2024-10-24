#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 21 00:21:21 2023

@author: prai
"""

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
from .viscousburgers1D import Grid1d, SimulationViscousBurgers

import matplotlib.pyplot as plt

def visualization(x,U):
    fig = plt.figure()
    plt.plot(x,U.T)
    plt.show()
    #plt.pause(0.1)
    # plt.close()


def paramRange_1d():
    tmin = 0.; tmax = 5.
    xmin = -3; xmax = 5 #Need to caliberate
    ymin = 0.5; ymax = 3.0 #y parameter, refer to Ehrlacher et al. 2020
    numin= 5.e-5; numax = 0.1 #Viscous Burgers
    
    return [(xmin, xmax), (numin, numax), (ymin, ymax), (tmin,tmax)]

def solve(p,grid):
    p = p.numpy()
    nu = p[1]; y = p[2]; t = p[3]
    cfl = 0.1
    s = SimulationViscousBurgers(grid)
    s.init_cond(y)
    s.evolve(nu, cfl, t, dovis=0)
    return s.grid.data["u"]

if __name__ == "__main__":

    # Parse
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', type=int, default=10, help='number of parameters')
    args = parser.parse_args()

    # Data are always generated on cpu
    # The device is then changed during the import
    device = torch.device('cpu')
    
    xmin = -3; xmax = 5; nx = 1000; ng = 2
    
    # Spatial coordinates
    # x = torch.linspace(start=xmin, end=xmax, steps=nx, dtype=dtype, device=device)
    # x = (torch.arange(nx) + 0.5)*dx + xmin
    grid = Grid1d(nx, ng=ng, vars=["u"], xmin=xmin, xmax=xmax)
    x    = grid.x
    
    # Parameters
    paramRange = paramRange_1d()
    paramDomain = ParameterDomain(paramRange)
    nparam = args.n
    samplingStrategy = SamplingRandom(paramDomain, nparam)

    for i in range(3):    
        params = torch.tensor([p for p in samplingStrategy], dtype=dtype, device=device)

        # Snapshots
        print('Beginning computation of snapshots.',i)
        tic = time.time()
        fields = []
        for p in params:
            u = solve(p,grid)
            fields.append( torch.from_numpy(u) )
            # visualization(x, u)
        toc = time.time()
        print('End computation of snapshots. Took '+str(toc-tic)+' sec.')

        # Save
        rootdir = check_create_dir(data_dir+'ViscousBurger1d/'+str(nparam)+'/'+str(i)+'/')
        torch.save(params, rootdir+'params')
        # torch.save(x, rootdir+'points')
        torch.save(x, rootdir+'points')
        torch.save(fields, rootdir+'fields')
        with open(rootdir+"uuid.txt", "w") as uuid_file: # save unique identifier
            uuid_file.write(uuid.uuid4().hex)

