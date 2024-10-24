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

def fromIndexToCell(index, nmail):
    icell = index[1]*nmail[0]+index[0]
    return icell

def getCellIndex(icell, nmail):
    index = [icell % nmail[0], icell // nmail[0]]
    return index 

def spatial_coordinates_2d(xmin, xmax, nmail):
    Ndim  = nmail.shape[0]
    ncell = torch.prod(nmail)
    grid  = [torch.linspace(xmin[idim], xmax[idim], nmail[idim]+1) for idim in range(Ndim)]
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

def visualization(grid, dh, U):

    fig = plt.figure()
    #ax = plt.axes(projection='3d')
    x = grid[0][:-1]+0.5*dh[0]
    y = grid[1][:-1]+0.5*dh[1]
    X,Y = np.meshgrid(x.numpy(), y.numpy())
    #ax.plot_wireframe(X,Y,U.numpy().T, cmap='hsv')
    #ax.plot_surface(X,Y,U.numpy().T, cmap='hot')
    
    plt.pcolor(X,Y,U.numpy().T,cmap='jet',shading='auto')
    plt.colorbar()
    plt.show()
    #plt.pause(0.1)
    plt.close()

def visualization_mayavi(grid,dh,U):
    fig = mlab.figure()
    x = grid[0][:-1] + 0.5 * dh[0]
    y = grid[1][:-1] + 0.5 * dh[1]
    X, Y = np.meshgrid(x.numpy(), y.numpy(),indexing='ij')
    # mlab.surf(X,Y,U.numpy().T,  warp_scale='auto')
    # mlab.show()
    mlab.imshow(U.numpy())
    mlab.colorbar(orientation='vertical')
    mlab.view(0,0)
    mlab.show()

def animation_frames_2D(grid,dh,imgs):
    #import matplotlib.animation as animation
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
    # s = mlab.surf(X, Y, imgs[0].numpy().T,  warp_scale='auto')
    # mlab.colorbar(orientation='vertical')
    
    s = mlab.imshow(imgs[0].numpy().T)
    mlab.colorbar(orientation='vertical')
    mlab.view(0,0)
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

def periodic_BD(z):
    z[0,:]  = z[-2,:]
    z[-1,:] = z[1,:]
    z[:,0] = z[:,-2]
    z[:,-1]= z[:,1]

# def myperiodic_BD(z):
#     nx = z.shape[0]-2
#     ny = z.shape[1]-2
#     z[0,:] = z[nx,:]
#     z[nx+1,:] = z[1,:]
#     z[:,0] = z[:,ny]
#     z[:,ny+1]= z[:,1]

def Ineedtosave(n, t, nbsavemax,tmax):
    res = (np.floor(nbsavemax*t/tmax)>= n+1)
    return res
def calculE(rho, Lap, Grad):
    rhonew = 1.0 -rho
    phi,_ = torch.solve(rhonew[:-1],Lap[:-1,:-1])
    phip = torch.zeros((1,1))
    phinew = torch.vstack((phi,phip))
    return -torch.matmul(Grad,phinew)

def getEgradv(E,xmin, nmail, dh):
    ncell = torch.prod(nmail)
    Egradv = torch.zeros((ncell, ncell))
    for j in range(nmail[1]):
        for i in range(nmail[0]):
            xi = xmin[0] + (i + 0.5) * dh[0]
            vj = xmin[1] + (j + 0.5) * dh[1]
            icell = fromIndexToCell([i, j], nmail)
            Egradv[icell, icell] = 0

            icellplus = fromIndexToCell([i, j + 1], nmail) if j < nmail[1] - 1 else fromIndexToCell([i, 0], nmail)
            icellminus = fromIndexToCell([i, j - 1], nmail) if j > 0 else fromIndexToCell([i, nmail[1] - 1], nmail)

            Egradv[icell, icellplus] = 1.0 / (2 * dh[1]) * E[i]
            Egradv[icell, icellminus] = -1.0 / (2 * dh[1]) * E[i]
    return Egradv
def getVgradx(xmin,nmail, dh):
    ncell = torch.prod(nmail)
    Vgradx = torch.zeros((ncell, ncell))
    for j in range(nmail[1]):
        for i in range(nmail[0]):
            xi = xmin[0] + (i + 0.5) * dh[0]
            vj = xmin[1] + (j + 0.5) * dh[1]
            icell = fromIndexToCell([i, j], nmail)
            Vgradx[icell, icell] = 0
            icellplus = fromIndexToCell([i + 1, j], nmail) if i < nmail[0] - 1 else fromIndexToCell([0, j], nmail)
            icellminus = fromIndexToCell([i - 1, j], nmail) if i > 0 else fromIndexToCell([nmail[0] - 1, j], nmail)
            Vgradx[icell, icellplus] = 1.0 / (2 * dh[0]) * vj
            Vgradx[icell, icellminus] = -1.0 / (2 * dh[0]) * vj
    return Vgradx

def Upwind_flux(u,v, E, nmail):
    nx = nmail[0]
    ny = nmail[1]
    F=torch.zeros((nx+1,ny)) # fluxes in the $x$ direction
    G=torch.zeros((nx,ny+1)) # fluxes in the $y$ direction

    for i in range(nmail[0]+1):
        for j in range(nmail[1]):
            ul = u[i,j+1]
            ur = u[i+1,j+1]
            vl = v[i,j+1]
            vr = v[i+1,j+1]
            F[i,j] = max(vl,0)*ul + min(vr,0)*ur 
    for i in range(nmail[0]):
        for j in range(nmail[1]+1):
            ul = u[i+1,j]
            ur = u[i+1,j+1]
            El = E[i+1,j]
            Er = E[i+1,j+1]
            G[i,j]= max(El,0)*ul + min(Er,0)*ur 
    return F, G

def Upwind_flux_fast(u,v,E,nmail):
    # F=torch.zeros((nmail[0]+1,nmail[1])) # fluxes in the $x$ direction
    # G=torch.zeros((nmail[0],nmail[1]+1)) # fluxes in the $y$ direction

    # F= np.maximum(v[0:-1,1:-1] ,0)*u[0:-1,1:-1] - np.maximum(-v[1:,1:-1],0) *u[1:,1:-1]
    # G= np.maximum(E[1:-1,0:-1] ,0)*u[1:-1,0:-1] - np.maximum(-E[1:-1,1:],0) *u[1:-1,1:]

    F= np.maximum(v[0:-1,1:-1] ,0)*u[0:-1,1:-1] - np.maximum(-v[0:-1,1:-1],0) *u[1:,1:-1]
    G= np.maximum(E[1:-1,0:-1] ,0)*u[1:-1,0:-1] - np.maximum(-E[1:-1,0:-1],0) *u[1:-1,1:]

    return F, G

def FBM_flux(u,v,E,nmail,dt,dh):
    nx = nmail[0]
    ny = nmail[1]
    F=torch.zeros((nx+1,ny)) # fluxes in the $x$ direction
    G=torch.zeros((nx,ny+1)) # fluxes in the $y$ direction

    for i in range(nmail[0]+1):
        for j in range(nmail[1]):
            fi =   u[i,j+1]
            fip1 = u[i+1,j+1]
            if i >0 :
                fim1 = u[i-1,j+1]
            else:
                fim1 = u[nx-1,j+1]
            if i < nx:
                fip2 = u[i+2,j+1]
            else:
                fip2 = u[2,j+1]
            alpha = v[i,j+1]
            if alpha >0 :
                F[i,j] = alpha*fi*dt/dh[0] + (fip1-fim1)*alpha*dt/(4*dh[0])*(1-alpha*dt/dh[0])
            else:
                F[i,j] = alpha*fip1*dt/dh[0] + (fip2-fi)*np.abs(alpha)*dt/(4*dh[0])*(1-alpha*dt/dh[0])


    for i in range(nmail[0]):
        for j in range(nmail[1]+1):
            fi =   u[i+1,j]
            fip1 = u[i+1,j+1]
            if j>0:
                fim1 = u[i+1,j-1]
            else:
                fim1 = u[i+1,ny-1]
            if j< ny:
                fip2 = u[i+1,j+2]
            else:
                fip2 = u[i+1,2]
            
            alpha = E[i+1,j]
            if alpha >0 :
                G[i,j] = alpha*fi*dt/dh[1] + (fip1-fim1)*alpha*dt/(4*dh[1])*(1-alpha*dt/dh[1]) 
            else:
                G[i,j] = alpha*fip1*dt/dh[1] + (fip2-fi)*np.abs(alpha)*dt/(4*dh[1])*(1-alpha*dt/dh[1])
            
    return F, G
def FBM_flux_fast(u,v,E,nmail,dt,dh):
    nx = nmail[0]
    ny = nmail[1]
    F=torch.zeros((nx+1,ny)) # fluxes in the $x$ direction
    G=torch.zeros((nx,ny+1)) # fluxes in the $y$ direction
    F[1:-1,:] = np.maximum(v[1:-2,1:-1],0)*( u[1:-2,1:-1]*dt/dh[0]  + (u[2:-1,1:-1]-u[0:-3,1:-1])*dt/(4*dh[0])*(1-v[1:-2,1:-1]*dt/dh[0]) ) + \
                np.minimum(v[1:-2,1:-1],0)*( u[2:-1,1:-1]*dt/dh[0]  - (u[3: ,1:-1] -u[1:-2,1:-1])*dt/(4*dh[0])*(1-v[1:-2,1:-1]*dt/dh[0]) )
    F[0,:]    = np.maximum(v[0,1:-1],0)*( u[0,1:-1]*dt/dh[0]  + (u[1,1:-1]- u[nx-1,1:-1])*dt/(4*dh[0])*(1-v[0,1:-1]*dt/dh[0]) ) + \
                np.minimum(v[0,1:-1],0)*( u[1,1:-1]*dt/dh[0]  - (u[2,1:-1] -u[0,1:-1])*dt/(4*dh[0])*(1-v[0,1:-1]*dt/dh[0]) )
    F[-1,:]   = np.maximum(v[nx,1:-1],0)*( u[nx,1:-1]*dt/dh[0]  + (u[nx+1,1:-1]- u[nx-1,1:-1])*dt/(4*dh[0])*(1-v[nx,1:-1]*dt/dh[0]) ) + \
                np.minimum(v[nx,1:-1],0)*( u[nx+1,1:-1]*dt/dh[0]  - (u[2,1:-1] -u[nx,1:-1])*dt/(4*dh[0])*(1-v[nx,1:-1]*dt/dh[0]) )
    #=========================================================================================================================================
    G[:,1:-1] = np.maximum(E[1:-1,1:-2],0)*( u[1:-1,1:-2]*dt/dh[1]  + (u[1:-1,2:-1]-u[1:-1,0:-3])*dt/(4*dh[1])*(1-E[1:-1,1:-2]*dt/dh[1]) ) + \
                np.minimum(E[1:-1,1:-2],0)*( u[1:-1,2:-1]*dt/dh[1]  - (u[1:-1,3: ] -u[1:-1,1:-2])*dt/(4*dh[1])*(1-E[1:-1,1:-2]*dt/dh[1]) )

    G[:,0]    = np.maximum(E[1:-1,0],0)*( u[1:-1,0]*dt/dh[1]  + (u[1:-1,1]- u[1:-1,ny-1])*dt/(4*dh[1])*(1-E[1:-1,0]*dt/dh[1]) ) + \
                np.minimum(E[1:-1,0],0)*( u[1:-1,1]*dt/dh[1]  - (u[1:-1,2] -u[1:-1,0])*dt/(4*dh[1])*(1-E[1:-1,0]*dt/dh[1]) )

    G[:,-1]   = np.maximum(E[1:-1,ny],0)*( u[1:-1,ny]*dt/dh[1]  + (u[1:-1,ny+1]- u[1:-1,ny-1])*dt/(4*dh[1])*(1-E[1:-1,ny]*dt/dh[1]) ) + \
                np.minimum(E[1:-1,ny],0)*( u[1:-1,ny+1]*dt/dh[1] -(u[1:-1,2] -u[1:-1,ny])*dt/(4*dh[1])*(1-E[1:-1,ny]*dt/dh[1]) )

    return F, G




def vlasov_poisson_1d_1v(params,xmin, grid, nmail, dh):
    ncell = torch.prod(nmail)
    # Initial condition
    U0 = torch.zeros((ncell,1))
    V0 = torch.zeros((ncell,1))
    # v0 = 3.5
    # beta = 1.e-3
    v0 = params[0]
    beta = params[1]

    #g = lambda x1, x2: (1+ beta*torch.cos(0.5*x1)) *1.0*x2**2/(1*torch.sqrt(2* torch.tensor(np.pi)))*(torch.exp(-0.5*(x2-v0)**2) )
    g = lambda x1, x2: (1+ beta*torch.cos(0.2*x1)) *1.0/(2*torch.sqrt(2* torch.tensor(np.pi)))*(torch.exp(-0.5*(x2-v0)**2) + torch.exp(-0.5*(x2+v0)**2))+1.e-6
    #g = lambda x1, x2: (1+ beta*torch.cos(0.5*x1)) *(1.0/(1*torch.sqrt(2* torch.tensor(np.pi)))*torch.exp(-0.5*(x2-v0)**2) )
    for i in range(nmail[0]):
        for j in range(nmail[1]):
            x = xmin[0] + (i+0.5)*dh[0]
            y = xmin[1] + (j+0.5)*dh[1]
            icell = fromIndexToCell([i, j], nmail)
            U0[icell] = g(x, y)
            V0[icell] = y
    # normalization
    #U0 = U0 / (torch.sum(U0)*torch.prod(dh))
    #print('check',torch.sum(U0)*torch.prod(dh))
    u = torch.zeros((nmail[0]+2,nmail[1]+2))
    v = torch.zeros((nmail[0]+2,nmail[1]+2))
    E = torch.zeros((nmail[0]+2,nmail[1]+2))

    u[1:-1,1:-1] = U0.reshape(nmail[1],nmail[0]).T
    #visualization(grid,dh, u[1:-1,1:-1])
    v[1:-1,1:-1] = V0.reshape(nmail[1],nmail[0]).T
    periodic_BD(v)
    #visualization(grid,dh, v[1:-1,1:-1])

    u_old = deepcopy(u)
    periodic_BD(u_old)

    Lap = torch.zeros((nmail[0],nmail[0])) 
    Grad = torch.zeros((nmail[0],nmail[0])) 

    for i in range(nmail[0]):
        Lap[i,i] = -2/dh[0]**2
        Grad[i,i] = 0
        mpx = i+1 if i < nmail[0]-1 else 0
        mmx = i-1 if i > 0 else nmail[0]-1
        Lap[i,mpx] = 1.0/dh[0]**2
        Lap[i,mmx] = 1.0/dh[0]**2
        Grad[i,mpx] = 1.0/(2*dh[0])
        Grad[i,mmx] = -1.0/(2*dh[0])


    # perform the solver implicit in time

    t = 0
    Tmax = 40
    dt = 0.002
    nt = int(Tmax/dt)
    nbsavemax = 400
    n = 0
    savet = np.zeros(nbsavemax)

    Ulist=[]
    tlist=[]
    Ulist.append(u)
    tlist.append(t)
    imgs = []
    for it in range(nt):

        # Evaluate E

        rho = torch.matmul(deepcopy(u_old[1:-1,1:-1]), dh[1]* torch.ones((nmail[1], 1)))
        E[1:-1,1:-1] = calculE(rho,Lap,Grad)
        periodic_BD(E)
        
        # Upwind scheme
        # F, G = Upwind_flux_fast(u_old,v,E,nmail)
        # u[1:-1,1:-1] = u_old[1:-1,1:-1]-dt/dh[0]*(F[1:,:]-F[0:-1,:]) -dt/dh[1]*(G[:,1:] - G[:,0:-1])


        # FBM scheme
        F, G = FBM_flux_fast(u_old,v,E,nmail,dt,dh)
        u[1:-1,1:-1] = u_old[1:-1,1:-1]-(F[1:,:]-F[0:-1,:]) -(G[:,1:] - G[:,0:-1])

        # Splitting scheme 


        # # dt/2
        # F, _ = Upwind_flux_fast(u_old,v,E,nmail)
        # u_old[1:-1,1:-1] = u_old[1:-1,1:-1]-1/2*dt/dh[0]*(F[1:,:]-F[0:-1,:])
        # periodic_BD(u_old)
        # # update E
        # rho = torch.matmul(u_old[1:-1,1:-1], dh[1]* torch.ones((nmail[1], 1)))
        # E[1:-1,1:-1] = calculE(rho,Lap,Grad)
        # periodic_BD(E)
        # # dt 
        # _, G = Upwind_flux_fast(u_old,v,E,nmail)
        # u_old[1:-1,1:-1] = u_old[1:-1,1:-1]-dt/dh[1]*(G[:,1:] - G[:,0:-1])
        # periodic_BD(u_old)
        # # dt/2
        # u[1:-1,1:-1] = u_old[1:-1,1:-1]-1/2*dt/dh[0]*(F[1:,:]-F[0:-1,:])



        # Method 3
        # u[1:-1,1:-1]= u_old[1:-1,1:-1]-\
        # dt/(2*dh[0])*(v[2:,1:-1]*u_old[2:,1:-1] - v[0:-2,1:-1]*u_old[0:-2,1:-1]) -\
        # dt/(2*dh[1])*(E[1:-1,2:]*u_old[1:-1,2:] - E[1:-1,0:-2]*u_old[1:-1,0:-2]) +\
        # dt*1/2*(u_old[2:,1:-1]-2*u_old[1:-1,1:-1]+ u_old[0:-2,1:-1])/dh[0] +\
        # dt*1/2*(u_old[1:-1,2:]-2*u_old[1:-1,1:-1]+ u_old[1:-1,0:-2])/dh[1]

        # Periodic boundary condition
        periodic_BD(u)
        u_old = deepcopy(u)
        t = t +dt
        
        if Ineedtosave(n,t,nbsavemax,Tmax):
            savet[n] = t
            n = n+1
            imgs.append(deepcopy(u[1:-1,1:-1]))

        #visualization(grid,dh,u[1:-1,1:-1])

    # normalization
    u[1:-1,1:-1] = u[1:-1,1:-1]/(torch.sum(u[1:-1,1:-1])*torch.prod(dh))

    #print('check', torch.sum(u[1:-1,1:-1])*torch.prod(dh) )
    #animation_mayavi_2D(grid,dh,imgs)
    #visualization(grid,dh,u[1:-1,1:-1])
    return u[1:-1,1:-1]
    
    



def params_range_2d():
    vmin = 2; vmax = 3.5
    bmin = 1.e-3; bmax = 5.e-3
    return [(vmin, vmax), (bmin, bmax)]

if __name__ == "__main__":
    """Generates Vlasov-poisson equation.
    """

    # Parse
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', type=int, default=10, help='number of parameters')
    args = parser.parse_args()

    # Data are always generated on cpu
    # The device is then changed during the import
    device = torch.device('cpu')


    xmin  = torch.tensor([0, -10])
    xmax  = torch.tensor([10*np.pi, 10])
    nmail = torch.tensor([64, 64])

    # Spatial coordinates
    grid, dh, X = spatial_coordinates_2d(xmin,xmax,nmail)


    # Parameters

    # params = [2.4, 0.001]
    # U = vlasov_poisson_1d_1v(params,xmin, grid, nmail, dh)
    
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
            U = vlasov_poisson_1d_1v(p,xmin, grid, nmail, dh)
            fields.append( U )
        toc = time.time()
        print('End computation of snapshots. Took '+str(toc-tic)+' sec.')

        # Save
        rootdir = check_create_dir(data_dir+'VlasovPoisson/'+str(nparam)+'/'+str(i)+'/')
        torch.save(params, rootdir+'params')
        torch.save(X, rootdir+'points')
        torch.save(fields, rootdir+'fields')
        with open(rootdir+"uuid.txt", "w") as uuid_file: # save unique identifier
            uuid_file.write(uuid.uuid4().hex)