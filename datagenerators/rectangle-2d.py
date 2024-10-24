# SPDX-FileCopyrightText: The sinkhorn-rom authors, see file AUTHORS
#
# SPDX-License-Identifier: GPL-3.0-or-later

import torch
import time
import uuid
import argparse
from scipy.stats import random_correlation

from .parameterDomain import ParameterDomain
from .sampling import *
from ..src.utils import check_create_dir
from ..src.config import data_dir, dtype

def rectange_2d(params, X):
    
    return torch.tensor([ 1.0/(params[0]*params[1])*(2*(x[0]-0.)<= params[0])*(2*(x[0]-0.)>= -params[0])*(2*(x[1]-0.)<= params[1])*(2*(x[1]-0.)>= -params[1]) for x in X ], dtype=dtype, device=device)+1.e-8

def spatial_coordinates_2d(nx=20, xmin=-4., xmax=4.):
    x = torch.linspace(start=xmin, end=xmax, steps=nx, dtype=dtype, device=device)
    return torch.cartesian_prod(x, x)

def params_range_2d():
    a_min = 1.0; a_max = 2.
    b_min = 1.0 ; b_max =2.
    return [(a_min, a_max), (b_min, b_max)]

if __name__ == "__main__":
    """Generates 2d-rectangles solution.
    """

    # Parse
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', type=int, default=500, help='number of parameters')
    args = parser.parse_args()

    # Data are always generated on cpu
    # The device is then changed during the import
    device = torch.device('cpu')

    # Spatial coordinates
    X = spatial_coordinates_2d()

    # Parameters

    paramRange = params_range_2d()
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
            fields.append( rectange_2d(p, X) ) 
        toc = time.time()
        print('End computation of snapshots. Took '+str(toc-tic)+' sec.')

        # Save
        rootdir = check_create_dir(data_dir+'Rectangle2d/'+str(nparam)+'/'+str(i)+'/')
        torch.save(params, rootdir+'params')
        torch.save(X, rootdir+'points')
        torch.save(fields, rootdir+'fields')
        with open(rootdir+"uuid.txt", "w") as uuid_file: # save unique identifier
            uuid_file.write(uuid.uuid4().hex)

