# SPDX-FileCopyrightText: The sinkhorn-rom authors, see file AUTHORS
#
# SPDX-License-Identifier: GPL-3.0-or-later

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

def Ineedtosave(n, t, nbsavemax,tmax):
    res = (np.floor(nbsavemax*t/tmax)>= n+1)
    return res



# def params_range_2d():
#     xmin = 0.4; xmax = 0.6
#     ymin = 0.4; ymax = 0.6
#     wmin = 0.1; wmax = 0.2
#     numin= 5.e-5; numax = 0.01
#     tmin = 0.5 ; tmax = 1
#     return [(xmin, xmax), (ymin, ymax), (wmin, wmax),(numin,numax),(tmin,tmax)]
#     #return [(xmin, xmax), (ymin, ymax), (wmin, wmax)]
def params_range_2d(num_rect=1):
    xmin = 2; xmax = 6
    ymin = 2; ymax = 6
    wmin = 1; wmax = 2
    numin= 5.e-5; numax = 0.1
    tmin = 0. ; tmax = 5.
    return num_rect*[(xmin, xmax), (ymin, ymax), (wmin, wmax)]+[(numin,numax),(tmin,tmax)]

def params_range_2d_two_bump():
    xmin1 = 2; xmax1 = 6
    ymin1 = 1; ymax1 = 2
    wmin1 = 0.5; wmax1 = 1
    xmin2 = 2; xmax2 = 6
    ymin2 = 5; ymax2 = 6
    wmin2 = 0.5; wmax2 = 1
    numin= 5.e-5; numax = 0.1
    tmin = 0. ; tmax = 3.
    return [(xmin1, xmax1), (ymin1, ymax1), (wmin1, wmax1)]+[(xmin2, xmax2), (ymin2, ymax2), (wmin2, wmax2)]+[(numin,numax),(tmin,tmax)]


def rectangle_2d(params, X):
    x0= params[0]
    y0= params[1]
    w = params[2]
    return torch.tensor([ 1.0/(w*w)*(x[0]>= x0-w/2)*(x[0]<=x0+w/2)*(x[1]>= y0-w/2)*(x[1]<=y0+w/2) for x in X ], dtype=dtype, device=device)+1.e-8

def mixed_rectangle_2d(params, X,num_rect=3):
    values = 1.e-8
    for i in range(num_rect):
        id = 3*i
        x0= params[id+0]
        y0= params[id+1]
        w = params[id+2]
        values += torch.tensor([ 1.0/(w*w)*(x[0]>= x0-w/2)*(x[0]<=x0+w/2)*(x[1]>= y0-w/2)*(x[1]<=y0+w/2) for x in X ], dtype=dtype, device=device)
    return values


def gaussian_2d(params, X):
    m = torch.tensor([params[0], params[1]], dtype=dtype, device=device)
    eigs = np.array([params[2], 2.-params[2]])
    # print(np.fabs(np.sum(eigs) - eigs.size)>0)
    C = torch.tensor(random_correlation.rvs(eigs, tol=1.e-6), dtype=dtype, device=device)
    A = 2*np.pi*torch.sqrt(torch.det(C))
    return A*torch.exp(torch.tensor([ -0.5*torch.matmul(x-m, torch.matmul(C, x-m)).item() for x in X ], dtype=dtype, device=device))+1.e-8

def solver_burger_2d(params,grid,nmail,dh, X):

    nu = params[-2]
    #dt = 0.0001
    #dt = 0.00002
    dt  = 0.001
    Tmax = params[-1]

    # nu = 0.01
    # dt = 0.0001
    # Tmax = 0.05


    nt = int(Tmax/dt)

    # initial condition
    u = torch.zeros((nmail[0]+2,nmail[1]+2))
    U = rectangle_2d(params,X)
    #print('mass',torch.sum(U))
    # normalization
    #U = U/(torch.sum(U)*torch.prod(dh))
    U = 100*U/torch.sum(U)

    u[1:-1,1:-1]= U.reshape(nmail[1],nmail[0]).T
    #print('check',torch.sum(deepcopy(u[1:-1,1:-1]).flatten()))
    u_old = deepcopy(u)
    
    #visualization(grid,dh,u[1:-1,1:-1])
    for n in range(nt):
        u[1:-1,1:-1]= u_old[1:-1,1:-1]-dt/dh[0]*(1/2*u_old[1:-1,1:-1]**2-1/2*u_old[0:-2,1:-1]**2) -\
            dt/dh[1]*(1/2*u_old[1:-1,1:-1]**2-1/2*u_old[1:-1,0:-2]**2) + \
            nu*dt/dh[0]**2*(u_old[2:,1:-1]-2*u_old[1:-1,1:-1]+ u_old[0:-2,1:-1]) +\
            nu*dt/dh[1]**2*(u_old[1:-1,2:]-2*u_old[1:-1,1:-1]+ u_old[1:-1,0:-2])

        # Periodic boundary condition
        u[0,:] = u[-2,:]
        u[-1,:]= u[1,:]
        u[:,0] = u[:,-2]
        u[:,-1]= u[:,1]
        
        u_old = deepcopy(u)
        #print('check',torch.sum(deepcopy(u[1:-1,1:-1]).flatten())* torch.prod(dh) )
        #print('check',torch.sum(deepcopy(u[1:-1,1:-1]).flatten()) )
    #visualization(grid,dh,u[1:-1,1:-1])
    #visualization(grid,dh,u[1:-1,1:-1]/torch.sum(u[1:-1,1:-1]))
    return u[1:-1,1:-1]/torch.sum(u[1:-1,1:-1])

if __name__ == "__main__":
    """Generates burger-2d equation.
    """

    # Parse
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', type=int, default=10, help='number of parameters')
    args = parser.parse_args()

    # Data are always generated on cpu
    # The device is then changed during the import
    device = torch.device('cpu')


    xmin  = torch.tensor([0, 0])
    xmax  = torch.tensor([10, 10])
    nmail = torch.tensor([64, 64])

    # Spatial coordinates
    grid, dh, X = spatial_coordinates_2d(xmin,xmax,nmail)
    # Parameters

    paramRange = params_range_2d(num_rect=1)
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
        rootdir = check_create_dir(data_dir+'Burger2d/'+str(nparam)+'/'+str(i)+'/')
        torch.save(params, rootdir+'params')
        torch.save(X, rootdir+'points')
        torch.save(fields, rootdir+'fields')
        with open(rootdir+"uuid.txt", "w") as uuid_file: # save unique identifier
            uuid_file.write(uuid.uuid4().hex)

