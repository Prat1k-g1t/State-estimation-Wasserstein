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

def gaussian_2d(params, X):
    m = torch.tensor([params[0], params[1]], dtype=dtype, device=device)
    eigs = np.array([params[2], 2.-params[2]])
    # print(np.fabs(np.sum(eigs) - eigs.size)>0)
    C = torch.tensor(random_correlation.rvs(eigs, tol=1.e-6), dtype=dtype, device=device)
    A = 2*np.pi*torch.sqrt(torch.det(C))
    return A*torch.exp(torch.tensor([ -0.5*torch.matmul(x-m, torch.matmul(C, x-m)).item() for x in X ], dtype=dtype, device=device))+1.e-8

def spatial_coordinates_2d(nx=64, xmin = -4., xmax=4.):
    x = torch.linspace(start=xmin, end=xmax, steps=nx, dtype=dtype, device=device)
    return torch.cartesian_prod(x, x)

def fromIndexToCell(index, nmail):
    icell = index[1]*nmail[0]+index[0]
    return icell

def getCellIndex(icell, nmail):
    index = [icell % nmail[0], icell // nmail[0]]
    return index 

def my_spatial_coordinates_2d(xmin, xmax, nmail):
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

    return points


def params_range_2d():
    xmin = -1.5; xmax = 1.5
    ymin = -1.5; ymax = 1.5
    lambda0_min = 0.5; lambda0_max = 1.5
    return [(xmin, xmax), (ymin, ymax), (lambda0_min, lambda0_max)]

if __name__ == "__main__":
    """Generates 2d-Gaussians.
    """

    # Parse
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', type=int, default=500, help='number of parameters')
    args = parser.parse_args()

    # Data are always generated on cpu
    # The device is then changed during the import
    device = torch.device('cpu')

    # Spatial coordinates
    #X = spatial_coordinates_2d()
    xmin  = torch.tensor([-5, -4])
    xmax  = torch.tensor([5, 4])
    nmail = torch.tensor([64, 64])
    X = my_spatial_coordinates_2d(xmin, xmax, nmail)
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
            fields.append( gaussian_2d(p, X) ) 
        toc = time.time()
        print('End computation of snapshots. Took '+str(toc-tic)+' sec.')

        # Save
        rootdir = check_create_dir(data_dir+'Gaussian2d/'+str(nparam)+'/'+str(i)+'/')
        torch.save(params, rootdir+'params')
        torch.save(X, rootdir+'points')
        torch.save(fields, rootdir+'fields')
        with open(rootdir+"uuid.txt", "w") as uuid_file: # save unique identifier
            uuid_file.write(uuid.uuid4().hex)

