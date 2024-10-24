# SPDX-FileCopyrightText: The sinkhorn-rom authors, see file AUTHORS
#
# SPDX-License-Identifier: GPL-3.0-or-later

import torch
import time
import uuid
import argparse
from scipy.stats import random_correlation
from copy import deepcopy
from mayavi import mlab
import matplotlib.animation as animation

from .parameterDomain import ParameterDomain
from .sampling import *
from ..src.utils import check_create_dir
from ..src.config import data_dir, dtype

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

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
    points = torch.zeros((ncell, 2))
    for j in range(nmail[1]):
        for i in range(nmail[0]):
            xi = xmin[0] + (i+0.5)*dh[0]
            vj = xmin[1] + (j+0.5)*dh[1]
            icell = fromIndexToCell([i, j], nmail)
            points[icell, 0] = xi
            points[icell, 1] = vj

    return grid, dh, points

def Ineedtosave(n, t, nbsavemax,tmax):
    res = (np.floor(nbsavemax*t/tmax)>= n+1)
    return res

def visualization(grid, dh, U):
    plt.figure()
    ax = plt.axes(projection='3d')
    x = grid[0][:-1] + 0.5 * dh[0]
    y = grid[1][:-1] + 0.5 * dh[1]
    X, Y = np.meshgrid(x.numpy(), y.numpy())
    # ax.plot_wireframe(X,Y,U.numpy(), cmap='hsv')
    myfig = ax.plot_surface(X,Y,U.numpy().T, cmap='jet')

    #plt.pcolor(X, Y, U.numpy().T, cmap='jet', shading='auto')
    plt.show()
    plt.colorbar(myfig)
    plt.close()

def visualization_mayavi(grid,dh,U):
    fig = mlab.figure()
    x = grid[0][:-1] + 0.5 * dh[0]
    y = grid[1][:-1] + 0.5 * dh[1]
    X, Y = np.meshgrid(x.numpy(), y.numpy(),indexing='ij')
    mlab.surf(X,Y,U.numpy().T,  warp_scale='auto')
    mlab.show()

# def animation_2D(grid,dh,imgs):
#     ### animation
#     from matplotlib.animation import FuncAnimation
#     x = grid[0][:-1] + 0.5 * dh[0]
#     y = grid[1][:-1] + 0.5 * dh[1]
#     X, Y = np.meshgrid(x.numpy(), y.numpy())
#     fig, ax = plt.subplots(subplot_kw=dict(projection='3d'), constrained_layout=True)

#     surf = ax.plot_surface(X,Y, imgs[0].numpy().T, cmap='viridis')

#     ax.autoscale(False)

#     def update(i):
#         surf.remove()
#         surf = ax.plot_surface(X,Y, imgs[i].numpy().T, cmap='viridis')
        

#         return [surf]

#     anim = FuncAnimation(fig, update, frames=range(len(imgs)), interval=2, repeat_delay=10)
#     ax.set(xlabel='x ', ylabel='y', zlabel='z')
#     #anim.save('movie.mp4')
#     plt.show()
def animation_frames_2D(grid,dh,imgs):
    import matplotlib.animation as animation
    x = grid[0][:-1] + 0.5 * dh[0]
    y = grid[1][:-1] + 0.5 * dh[1]
    X, Y = np.meshgrid(x.numpy(), y.numpy())
    frames = [] # for storing the generated images
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    for i in range(len(imgs)):
        #frames.append([ plt.pcolor(X, Y, imgs[i].numpy().T,shading='auto', cmap='jet', animated = True)])
        frames.append([ ax.plot_surface(X,Y,imgs[i].numpy().T, cmap='jet',animated = True)])

    ani = animation.ArtistAnimation(fig, frames, interval=2,repeat_delay=10)
    ani.save('movie_frames.mp4')
    plt.show()

def animation_mayavi_2D(grid,dh,imgs):
    import os
    # Output path for you animation images
    out_path = 'results/'
    os.makedirs(out_path,exist_ok=True)
    out_path = os.path.abspath(out_path)
    fps = 20
    prefix = 'ani'
    ext = '.png'

    x = grid[0][:-1] + 0.5 * dh[0]
    y = grid[1][:-1] + 0.5 * dh[1]
    X, Y = np.meshgrid(x.numpy(), y.numpy(),indexing='ij')
    s = mlab.surf(X, Y, imgs[0].numpy().T,  warp_scale='auto')
    padding = len(str(len(imgs)))
    @mlab.animate(delay=500)
    def anim():
        for i in range(len(imgs)):
            s.mlab_source.scalars = imgs[i].numpy()

            # create zeros for padding index positions for organization
            zeros = '0'*(padding - len(str(i)))
            # concate filename with zero padded index number as suffix
            filename = os.path.join(out_path, '{}_{}{}{}'.format(prefix, zeros, i, ext))
            mlab.savefig(filename=filename)
            yield

    anim()
    #mlab.view(distance=40)
    mlab.show()

    import subprocess
    ffmpeg_fname = os.path.join(out_path, '{}_%0{}d{}'.format(prefix, padding, ext))
    cmd = 'ffmpeg -f image2 -r {} -i {} -vcodec mpeg4 -y {}.mp4'.format(fps,ffmpeg_fname,prefix)

    print(cmd)
    subprocess.check_output(['bash','-c', cmd])

def periodic_boundary_condition(U):
    U[0,:,0]= U[-2,:,0]
    U[0,:,1]= U[-2,:,1]
    U[0,:,2]= U[-2,:,2]
    U[:,0,0]= U[:,-2,0]
    U[:,0,1]= U[:,-2,1]
    U[:,0,2]= U[:,-2,2]

    U[-1,:,0]= U[1,:,0]
    U[-1,:,1]= U[1,:,1]
    U[-1,:,2]= U[1,:,2]

    U[:,-1,0]= U[:,1,0]
    U[:,-1,1]= U[:,1,1]
    U[:,-1,2]= U[:,1,2]
def Neumann_boundary_condition(U):
    U[0,:,0]= U[1,:,0]
    U[0,:,1]= U[1,:,1]
    U[0,:,2]= U[1,:,2]
    U[:,0,0]= U[:,1,0]
    U[:,0,1]= U[:,1,1]
    U[:,0,2]= U[:,1,2]

    U[-1,:,0]= U[-2,:,0]
    U[-1,:,1]= U[-2,:,1]
    U[-1,:,2]= U[-2,:,2]

    U[:,-1,0]= U[:,-2,0]
    U[:,-1,1]= U[:,-2,1]
    U[:,-1,2]= U[:,-2,2]    
def periodic_BD(z):
    z[0,:]  = z[-2,:]
    z[-1,:] = z[1,:]
    z[:,0] = z[:,-2]
    z[:,-1]= z[:,1]
    
def Neumann_BD(z):
    z[0,:]  = z[1,:]
    z[-1,:] = z[-2,:]
    z[:,0] = z[:,1]
    z[:,-1]= z[:,-2]

def ES1_flux(U,z,nmail):
    nx = nmail[0]
    ny = nmail[1]
    F=torch.zeros((nx+2,ny+1,3)) # fluxes in the $x$ direction
    G=torch.zeros((nx+1,ny+2,3)) # fluxes in the $y$ direction
    #Compute fluxes in x-direction
    ni=1
    nj=0
    amax=0
    for i in range(1,nmail[0]+2):
        for j in range(1,nmail[1]+1):
            hl = U[i-1,j,0]
            hr = U[i,j,0]
            if torch.abs(hl) >= 1.e-4:
                ul = U[i-1,j,1]/hl
                vl = U[i-1,j,2]/hl
            else:
                ul = torch.zeros(1)
                vl = torch.zeros(1)
            if torch.abs(hr)>=1.e-4:
                ur = U[i,j,1]/hr
                vr = U[i,j,2]/hr
            else:
                ur = torch.zeros(1)
                vr = torch.zeros(1)
            zl = z[i-1,j]
            zr = z[i,j]
            dz=zr-zl
            F[i,j,:], a = ES1_solver(hl,hr,ul,ur,vl,vr,dz,ni,nj)
            amax=max([a, amax])
    # Compute fluxes in y-direction
    ni=0
    nj= 1
    for j in range(1,nmail[1]+2):
        for i in range(1,nmail[0]+1):
            hl = U[i,j-1,0]
            hr = U[i,j,0]
            if torch.abs(hl) >= 1.e-4:
                ul = U[i,j-1,1]/hl
                vl = U[i,j-1,2]/hl
            else:
                ul = torch.zeros(1)
                vl = torch.zeros(1)
            if torch.abs(hr) >= 1.e-4:
                ur = U[i,j,1]/hr
                vr = U[i,j,2]/hr
            else:
                ur = torch.zeros(1)
                vr = torch.zeros(1)
            
            zl = z[i,j-1]
            zr = z[i,j]
            dz=zr-zl
            G[i,j,:], a = ES1_solver(hl,hr,ul,ur,vl,vr,dz,ni,nj)
            amax=max([a, amax])
    return F, G, amax
def ES1_solver(hl,hr,ul,ur,vl,vr,dz,ni,nj):
    grav = 9.806
    uAvg=(ul+ur)/2.0
    vAvg=(vl+vr)/2.0
    upAvg= uAvg*ni+vAvg*nj
    hAvg=(hl+hr)/2.0
    hSqAvg = (hl**2 + hr**2)/2.0

    FEC= torch.tensor([[hAvg*upAvg], [hAvg*uAvg*upAvg+0.5*grav*hSqAvg*ni], [hAvg*vAvg*upAvg+ 0.5*grav*hSqAvg*nj]])

    cAvg=torch.sqrt(grav*hAvg)
    #M=math.sqrt(uAvg**2+vAvg**2)/cAvg
    R=1.0/np.sqrt(2*grav)*torch.tensor([[1, 0, 1] ,[uAvg-cAvg*ni, -nj*cAvg, uAvg+cAvg*ni], [vAvg-cAvg*nj, ni*cAvg, vAvg+cAvg*nj]])
    ld1=upAvg-cAvg;
    ld2=upAvg;
    ld3=upAvg+cAvg;
    A=torch.diag(torch.tensor([torch.abs(ld1), torch.abs(ld2), torch.abs(ld3)]))
    dh=hr-hl
    dSu= ur**2-ul**2
    dSv= vr**2-vl**2
    Vm=torch.tensor([[grav*dh+grav*dz-1/2*(dSu+dSv)], [ur-ul], [vr-vl]])

    DES1= torch.mm(torch.mm(R,A), torch.transpose(R,0,1))
    F= FEC-1/2* torch.mm(DES1,Vm)
    amax=torch.abs(upAvg)+cAvg 
    return F.flatten(), amax
def ES1_topography(U,z,dh,nmail):
    grav = 9.806
    h = U[:,:,0]
    S = torch.zeros((nmail[0]+2,nmail[1]+2,3))
    for i in range(1,nmail[0]+1):
        for j in range(1, nmail[1]+1):
            S[i,j,0] = 0
            S[i,j,1] = grav/(2*dh[0])*( (h[i,j]+h[i+1,j])/2*(z[i+1,j]-z[i,j]) + (h[i,j]+h[i-1,j])/2*(z[i,j]-z[i-1,j]) )
            S[i,j,2] = grav/(2*dh[1])*( (h[i,j]+h[i,j+1])/2*(z[i,j+1]-z[i,j]) + (h[i,j]+h[i,j-1])/2*(z[i,j]-z[i,j-1]) )
    return S

def mixed_gaussians_2d(params, X,num_gaussians=3):
    #print('params',params)
    values = 0.3
    for i in range(num_gaussians):
        id = 4*i
        mu_x= params[id+0]
        mu_y= params[id+1]
        sig_x = params[id+2]
        sig_y = params[id+3]
        values += torch.tensor([torch.exp(-sig_x*(x[0]-mu_x)**2-sig_y*(x[1]-mu_y)**2)  for x in X ], dtype=dtype, device=device)
    return values

def solver_shallow_water_2d(params, grid, nmail, dh,X):
    # initial condition 
    h = torch.zeros((nmail[0]+2,nmail[1]+2))
    u = torch.zeros((nmail[0]+2,nmail[1]+2))
    v = torch.zeros((nmail[0]+2,nmail[1]+2))
    z = torch.zeros((nmail[0]+2,nmail[1]+2))

    # x = grid[0][:-1] + 0.5 * dh[0]
    # y = grid[1][:-1] + 0.5 * dh[1]
    # for  i in range(1,nmail[0]+1):
    #     for j in range(1,nmail[1]+1):
    #         h[i,j]= 0.3+ 0.7*torch.exp(-params[2]*(x[i-1]-params[0])**2-params[3]*(y[j-1]-params[1])**2)
    H = mixed_gaussians_2d(params,X,num_gaussians=3)
    H = 100*H/torch.sum(H)
    h[1:-1,1:-1]= H.reshape(nmail[1],nmail[0]).T
    # print('I0', torch.sum(h[1:-1,1:-1])*torch.prod(dh))
    # visualization(grid,dh,h[1:-1,1:-1])

    # normalize = torch.sum(h[1:-1,1:-1])*torch.prod(dh)
    # h[1:-1,1:-1] = h[1:-1,1:-1]/normalize

    # print('I0=', torch.sum(h[1:-1,1:-1]))
    # visualization(grid,dh,h[1:-1,1:-1])

    U = torch.zeros((nmail[0]+2,nmail[1]+2, 3))
    U[:,:,0]= h
    U[:,:,1]= h*u
    U[:,:,2]= h*v
    periodic_boundary_condition(U)
    periodic_BD(z)

    dt = 0.01
    Tmax = params[-1]
    nt = int(Tmax/dt)



    imgs = []
    U0= deepcopy(U)
    for iter in range(nt):
        F,G ,amx = ES1_flux(U,z,nmail)
        S = ES1_topography(U,z,dh,nmail)
        #U(I,J,K)=U(I,J,K)-dt/dx*(F(I+1,J,K)-F(I,J,K))-dt/dy*(G(I,J+1,K)-G(I,J,K))
        U[1:-1,1:-1,:] = U[1:-1,1:-1,:] -dt/dh[0]*(F[2:,1:,:]-F[1:-1,1:,:])-dt/dh[1]*(G[1:,2:,:]-G[1:,1:-1,:])- dt*S[1:-1,1:-1,:]
        periodic_boundary_condition(U)
        #print('check=', torch.sum(U[1:-1,1:-1,0])*torch.prod(dh))
        # if iter % 4 ==0:
        #     imgs.append(deepcopy(U[1:-1,1:-1,0]))

    #visualization(grid,dh,U[1:-1,1:-1,0])
    
    return U[1:-1,1:-1,0]/torch.sum(U[1:-1,1:-1,0])
    
    



# def params_range_2d():
#     xmin = 0.4; xmax = 0.6
#     ymin = 0.4; ymax = 0.6
#     bmin = 100; bmax = 150
#     cmin = 100; cmax = 150 
#     tmin = 0.; tmax = 0.05

#     return [(xmin, xmax), (ymin, ymax),(bmin,bmax),(cmin,cmax),(tmin,tmax)]

def params_range_2d(num_gaussians=3):
    xmin = 4; xmax = 6
    ymin = 4; ymax = 6
    bmin = 5; bmax = 10
    cmin = 5; cmax = 10
    tmin = 0.; tmax = 4

    return num_gaussians*[(xmin, xmax), (ymin, ymax),(bmin,bmax),(cmin,cmax)] + [(tmin,tmax)]


if __name__ == "__main__":
    """Generates shallow water equation.
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

    # params = 
    # U = solver_shallow_water_2d(params,xmin, grid, nmail, dh)
    
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
            U = solver_shallow_water_2d(p,grid, nmail, dh,X)
            fields.append( U )
        toc = time.time()
        print('End computation of snapshots. Took '+str(toc-tic)+' sec.')

        # Save
        rootdir = check_create_dir(data_dir+'Shallow_Water2d/'+str(nparam)+'/'+str(i)+'/')
        torch.save(params, rootdir+'params')
        torch.save(X, rootdir+'points')
        torch.save(fields, rootdir+'fields')
        with open(rootdir+"uuid.txt", "w") as uuid_file: # save unique identifier
            uuid_file.write(uuid.uuid4().hex)