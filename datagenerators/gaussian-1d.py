# SPDX-FileCopyrightText: The sinkhorn-rom authors, see file AUTHORS
#
# SPDX-License-Identifier: GPL-3.0-or-later

import torch
import time
import uuid
import argparse

from .parameterDomain import ParameterDomain
from .sampling import *
from ..src.utils import check_create_dir
from ..src.config import data_dir, dtype
from torch.autograd import Variable
def gaussian_1d(params, X):
    return torch.exp(-0.5*(X-params[0])**2/params[1]**2)/(params[1]*np.sqrt(2*np.pi))+1.e-8

def spatial_coordinates_1d(nx=500, xmin = -4., xmax=4.):
    return torch.linspace(start=xmin, end=xmax, steps=nx, dtype=dtype, device=device)

def params_range_1d():
    # xmin = -1.5; xmax = 1.5
    xmin = -4.; xmax = 4.
    sigma_min = 0.5; sigma_max = 1.
    return [(xmin, xmax), (sigma_min, sigma_max)]

if __name__ == "__main__":
    """Generates 1d-Gaussians.
    """

    # Parse
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', type=int, default=500, help='number of parameters')
    args = parser.parse_args()

    # Data are always generated on cpu
    # The device is then changed during the import
    device = torch.device('cpu')

    # Spatial coordinates
    X = spatial_coordinates_1d()

    # Parameters
    paramRange = params_range_1d()
    paramDomain = ParameterDomain(paramRange)
    nparam = args.n
    samplingStrategy = SamplingRandom(paramDomain, nparam)

    # Generate 3 different data sets
    for i in range(3):
        params = torch.tensor([p for p in samplingStrategy], dtype=dtype, device=device)
        # print('params',params)

        # Snapshots
        print('Beginning computation of snapshots.')
        tic = time.time()
        fields = []
        for p in params:
            fields.append( gaussian_1d(p, X) ) 
        toc = time.time()
        print('End computation of snapshots. Took '+str(toc-tic)+' sec.')

        # Save
        rootdir = check_create_dir(data_dir+'Gaussian1d/'+str(nparam)+'/'+str(i)+'/')
        torch.save(params, rootdir+'params')
        torch.save(X.unsqueeze(1), rootdir+'points')
        torch.save(fields, rootdir+'fields')
        with open(rootdir+"uuid.txt", "w") as uuid_file: # save unique identifier
            uuid_file.write(uuid.uuid4().hex)


