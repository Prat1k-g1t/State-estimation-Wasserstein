#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 10:52:34 2024

@author: prai
"""

from scipy.special import logsumexp
import numpy as np
from typing import List
from tqdm import tqdm
from geomloss import SamplesLoss
import time
import copy
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.optim.lr_scheduler import MultiStepLR,ExponentialLR, StepLR
from ...config import use_pykeops, device, dtype
from geomloss import ImagesBarycenter

from geomloss.utils import log_dens, pyramid, upsample, softmin_grid, dimension
from geomloss.utils import softmin_grid as softmin
from geomloss.sinkhorn_divergence import epsilon_schedule,scaling_parameters,sinkhorn_cost,sinkhorn_loop
from geomloss.sinkhorn_images import kernel_truncation,extrapolate

from itertools import combinations

import cbx as cbx
from cbx.dynamics.cbo import CBO
from cbx.utils.objective_handling import cbx_objective_fh
# from cbx_torch_utils import flatten_parameters, get_param_properties, eval_losses, norm_torch, compute_consensus_torch, normal_torch, eval_acc, effective_sample_size
from cbx.scheduler import scheduler, multiply

import pyswarms as ps
from pyswarms.single.global_best import GlobalBestPSO
from pyswarms.utils.plotters import (plot_cost_history, plot_contour, plot_surface)
from pyswarms.utils.plotters.formatters import Mesher

from ...config import results_dir, fit_dir, use_pykeops, device
from ...utils import check_create_dir

from matplotlib import rc
rc('text', usetex=True)
import matplotlib.pyplot as plt

from scipy.linalg import toeplitz
import math
import imageio


from functools import partial

import sys

import pandas as pd

from typing import Callable, Any

from ..Evaluators.Barycenter import ImagesLoss, ImagesBarycenter_v2, mat2simplex, \
    projGraAdapSupp, ImagesBarycenter_1d
    
    
def convolution_kernel(field_coordinates, sensor_locations):
    
    field = field_coordinates.squeeze()
    
    tmp_field = field
    tmp_sensor_locations = sensor_locations
    
    variance = (field[-1]-field[0])*0.03
    
    if sensor_locations.ndim == 3:
        
        tmp_sensor_locations = torch.tensor(tmp_sensor_locations, device=device)
    
        M = sensor_locations.shape[0]
        N = sensor_locations.shape[1]
        d = sensor_locations.shape[-1]

        tmp_field = tmp_field.repeat(M,N,d,1)
                    
        tmp_sensor_locations = tmp_sensor_locations.reshape(M,N,d,1)
        
        gaussian = lambda x : (torch.exp(-0.5*((tmp_field-x)/variance)**2)/((variance*(2*math.pi)**0.5)))    
        
        ker = gaussian(tmp_sensor_locations)
        
        cell_indices = []
        
    # elif sensor_locations.ndim == 2:
        
    #     tmp_sensor_locations = torch.tensor(tmp_sensor_locations, device=device)
    
    #     N = sensor_locations.shape[0]
    #     d = sensor_locations.shape[-1]

    #     tmp_field = tmp_field.repeat(N,d,1)
                    
    #     tmp_sensor_locations = tmp_sensor_locations.reshape(N,d,1)
        
    #     gaussian = lambda x : (torch.exp(-0.5*((tmp_field-x)/variance)**2)/((variance*(2*math.pi)**0.5)))    
        
    #     ker = gaussian(tmp_sensor_locations)
        
    #     cell_indices = []
    
    else:
        
        # print('sensor_locations.dim()',sensor_locations.dim())
        
        h   = abs(field[0].item()-field[1].item())
        
        d = sensor_locations.shape[-1]
        
        cell_indices = []
        field_as_list            = field.tolist()
        sensor_locations_as_list = sensor_locations.tolist()[0]
        
        for i,j in enumerate(sensor_locations_as_list):
            
            if j < field[0].item():
                j = field[-1].item()
            elif j > field[-1].item():
                j = field[0].item()
            else:
                pass        
        
            # index = int(abs(field_as_list[0]-j)//h)
            index = int((j-field[0].item())//h)
                    
            cell_indices.append(index)
                    
        field = field.repeat(d,1)
        
        # tmp_sensor_locations = tmp_sensor_locations.reshape(d,1)
        
        gaussian = lambda x : (torch.exp(-0.5*((field-x)/variance)**2)/((variance*(2*math.pi)**0.5)))    
        
        ker = gaussian(sensor_locations.T)
    
    return ker, cell_indices

def compute_inf_sensor_placement_SGD(frames_dir, g1, g2, field_coordinates, sensor_locations):
    
    params_opt = {'optimizer': 'SGD', 'lr': 0.001, 'nmax': 100,'type_loss':'W2', 'gamma':1, 'momentum': 0.5}
    
    sensor_locations = torch.nn.Parameter(sensor_locations)
    
    gamma = params_opt['gamma']
    niter = params_opt['nmax']
    momentum = params_opt['momentum']
    
    def get_optimizer(params):
        if params_opt['optimizer'] == 'Adam':
            return torch.optim.Adam([params], lr=params_opt['lr'], maximize=True)
        # else:
        #     return torch.optim.SGD(params, lr=params_opt['lr'], momentum=0.9, maximize=True)
        
        elif params_opt['optimizer'] == 'SGD':
            return torch.optim.SGD([params], lr=params_opt['lr'], nesterov=True,
                                   momentum=momentum, maximize=True)

    field = field_coordinates.squeeze()
    
    optimizer = get_optimizer(sensor_locations)
    scheduler = ExponentialLR(optimizer, gamma=0.99)
    
    # ker, _ = self.convolution_kernel_v2(self.field_coordinates, sensor_locations)
    
    evolution = {'loss':[],'weights1':[],'weights2':[], 'sup_stability_const':[]}
    
    for iter in tqdm(range(niter)):
        
        # evolution['weights1'].append(params[0].detach().clone())
        # evolution['weights2'].append(params[1].detach().clone())
        
        optimizer.zero_grad()
        
        ker, cell_indices  = convolution_kernel(field_coordinates, sensor_locations)
        
        observations1 = torch.matmul(ker, g1.squeeze())
        observations2 = torch.matmul(ker, g2.squeeze())
        
        loss = torch.linalg.norm((observations1-observations2), axis=-1)
        
        loss.backward()
        optimizer.step()
        # scheduler.step()
        
        evolution['loss'].append(loss.item())
        evolution['sup_stability_const'].append(loss.item())
        
        plot_frames(frames_dir, g1, g2, sensor_locations, field, loss, ker, cell_indices, iter)
            
    ######## Plotting barycenters DEBUG ########
    # fig, axs = plt.subplots(1, sharex=True, sharey=True)
    
    # for i in range(self.measures.shape[1]): 
    #     axs.plot(self.field_coordinates.cpu().numpy(), torch.squeeze(self.measures.cpu())[i].numpy())                
    
    # axs.plot(self.field_coordinates.cpu().numpy(), bary1.cpu().detach().squeeze().numpy())
    # axs.plot(self.field_coordinates.cpu().numpy(), bary2.cpu().detach().squeeze().numpy())
    
    # axs.set_xlabel('x')
    # axs.set_title(r'Barycenters')
    
    # # axs[1].plot(self.field_coordinates.cpu().numpy(), observation_bary1.cpu().detach().squeeze().numpy())
    # # axs[1].plot(self.field_coordinates.cpu().numpy(), observation_bary2.cpu().detach().squeeze().numpy())
    # # axs[1].set_title(r'Observations')
    
    # fig.savefig(self.frames_dir+'barycenter_plot_{}.png'.format(self.iter))
    
    # plt.close()
    
    ############################################
    
    # print('sensor_locations', sensor_locations)
    
    plot_gif(frames_dir, iter)
    
    return sensor_locations.detach().clone(), np.array(evolution['loss']), ker, cell_indices

def compute_inf_sensor_placement_CBO(frames_dir, g1, g2, field, sensor_locations):
    
    conf = {'alpha': 50.0,
        'dt': 0.1,
        # 'max_time': 5.0,
        'sigma': 8, #self.k*(3.0+self.variance),
        'lamda': 1.0,
        # 'batch_args':{
        # 'batch_size':200,
        # 'batch_partial': False},
        'd': sensor_locations.shape[-1],
        'diff_tol': 1e-06,
        'energy_tol': 1e-06,
        'max_it': 500,
        'N': 100,
        'M': 1,
        'track_list': ['update_norm', 'energy','x', 'consensus', 'drift'],
        'resampling': False}
    
    # x = cbx.utils.init_particles(shape=(conf['M'], conf['N'], conf['d']),
    #                               x_min=field[0].item()*0.9, x_max = field[-1].item()*0.9)
    
    x = cbx.utils.init_particles(shape=(conf['M'], conf['N'], conf['d']),
                                  x_min=sensor_locations.squeeze()[0].item()*0.99, x_max = sensor_locations.squeeze()[-1].item()*0.99)
    
    def f(sensor_locations):
        
        ker, _ = convolution_kernel(field, sensor_locations)
        
        observations1 = torch.torch.matmul(ker,g1.squeeze())
        observations2 = torch.torch.matmul(ker,g2.squeeze())
        
        return 1./torch.linalg.norm((observations1-observations2), axis=-1).cpu().numpy()
    
    # Run the CBO algorithm
    it = 0
    inf_loss=np.zeros(conf['max_it'])
    
    dyn   = CBO(f, x=x, f_dim='3D', noise='anisotropic', **conf)
    
    sched = scheduler(dyn, [multiply(name='alpha', factor=1.1, maximum=1e15),
                        multiply(name='sigma', factor=1.005, maximum=9.0)
                        ])
    
    while not dyn.terminate():                                    
        
        print('Infimum iteration #',it)
        print('Loss', dyn.f_min)
        # print('Loss index', dyn.f_min_idx)
        # print('x',dyn.x)
        print('best_particles', dyn.best_particle[-1]) #, dyn.best_particle[1])
        inf_loss[it] = dyn.f_min[-1]
        
        best_particle = torch.tensor(dyn.best_particle[-1][None, :], dtype=dtype, device=device)
        
        ker, cell_indices = convolution_kernel(field, best_particle)
        
        plot_frames(frames_dir, g1, g2, best_particle, field, inf_loss, ker, cell_indices, it)
        
        it+=1
        
        dyn.step()
        sched.update()
        
    plot_gif(frames_dir, it-1)
        
    best_particle = torch.tensor(dyn.best_particle[-1][None, :], device=device)
    Cs            = dyn.f_min[-1]
    
    print('best_particle shape', best_particle)
    
    ker, cell_indices = convolution_kernel(field, best_particle)
    
    return best_particle, inf_loss, ker, cell_indices

def compute_inf_sensor_placement_PSO(frames_dir, g1, g2, field, sensor_locations):
    
    conf = {'c1': 0.5, 'c2': 0.3, 'w':0.9}
    
    no_of_sensors = sensor_locations.shape[-1]
    
    max_bound = field[-1].item()*0.9 * np.ones(no_of_sensors)
    min_bound = field[0].item()*0.9  * np.ones(no_of_sensors)
    bounds = (min_bound, max_bound)
    
    optimizer = ps.single.GlobalBestPSO(n_particles=10, dimensions=no_of_sensors, options=conf, bounds=bounds)
    
    def f(sensor_locations):
        
        sensor_locations = sensor_locations[None, :]
        
        #DEBUG:
        # print('sensor_locations.shape', sensor_locations.shape)
        
        ker, _ = convolution_kernel(field, sensor_locations)
        
        observations1 = torch.torch.matmul(ker,g1.squeeze())
        observations2 = torch.torch.matmul(ker,g2.squeeze())
        
        print('observations1.shape', observations1.shape)
        
        return 1./torch.linalg.norm((observations1-observations2), axis=-1).squeeze().cpu().numpy()
    
    cost, pos = optimizer.optimize(f, iters=1)
    
    # m = Mesher(func=f)
    
    # animation = plot_contour(pos_history=optimizer.pos_history,
    #                      mesher=m,
    #                      mark=(0,0))

    # animation.save('plot0.gif', writer='imagemagick', fps=10)
    # Image(url=frames_dir+'plot0.gif')

    
    # print('best_particle, cost types:', optimizer.pbest_pos)
    # print('best_particle, cost types:', type(optimizer.pos_history), type(optimizer.cost_history))
    # print('best_particle, cost shape:', np.array(optimizer.pos_history).shape, np.array(optimizer.cost_history).shape)
        
    # # best_particle = torch.tensor([torch.tensor(pos)], device=device)
    # pos           = torch.tensor(pos)
    # best_particle = pos[None, :].T
    # Cs            = cost
    
    pos = torch.tensor(pos[None, :], dtype=dtype, device=device)
    
    ker, cell_indices = convolution_kernel(field, pos)
    
    plot_frames(frames_dir, g1, g2, pos, field, cost, ker, cell_indices, 0)
    
    # return best_particle, Cs, ker, cell_indices
    
def define_measures():        
    
    field    = torch.linspace(-3,3,500).type(dtype).to(device)
    
    variance = (field[-1]-field[0])*0.01
    
    gaussian = lambda x : (torch.exp(-0.5*((field-x)/variance)**2)/((variance*(2*math.pi)**0.5)))
    
    loc1     = torch.tensor([-2.], dtype=dtype, device=device)
    
    g1       = gaussian(loc1)
    
    loc2     = torch.tensor([2.0], dtype=dtype, device=device)
    
    g2       = gaussian(loc2)
    
    #DEBUG:
    # print('g1', g1)
    # print('g2', g2)
    
    infimum_algo = 'PSO'
    
    no_of_sensors = 10
    
    sensor_locations = torch.linspace(field[0].item()*0.99, field[-1].item()*0.99, no_of_sensors).tolist(); arange = 'even'
    sensor_locations = torch.tensor([sensor_locations], dtype=dtype, device=device)
    
    frames_dir = check_create_dir(results_dir+'Test_optimal_sensor_placement/'+'Gaussian_measures/{}/'.format(infimum_algo))
    
    if infimum_algo == 'SGD':
        sensor_locations, L2, ker, cell_indices = compute_inf_sensor_placement_SGD(frames_dir, g1, g2, field, sensor_locations)
        
    elif infimum_algo == 'CBO':
        # sensor_locations, Cs, ker, cell_indices = compute_inf_x_Cs_CBO(measures, field_coordinates,
        #                                                                 params_opt, wts, no_of_sensors, bary1, bary2)
        
        sensor_locations, L2, ker, cell_indices = compute_inf_sensor_placement_CBO(frames_dir, g1, g2, field, sensor_locations)
    elif infimum_algo == 'PSO':
        compute_inf_sensor_placement_PSO(frames_dir, g1, g2, field, sensor_locations)
    
    
def plot_frames(frames_dir, g1, g2, sensor_locations, field, L2, ker, cell_indices, i):
    
    #Plotting:
        
    fig, axs = plt.subplots(2, sharex=False, sharey=False)
    
    sensor_placement = torch.zeros(field.shape[0])
    
    axs[0].set_title(r'Testing optimal sensor placement w/ evenly dist. of sensors')
    
    axs[0].plot(field.cpu().numpy(), g1.cpu().detach().squeeze().numpy())
    axs[0].plot(field.cpu().numpy(), g2.cpu().detach().squeeze().numpy())
    # axs[1].plot(field.cpu().numpy(), diff_bary)
    axs[0].plot(field.cpu().numpy(), sensor_placement.cpu().numpy(), markevery = cell_indices, ls = "", 
              marker="x", markerfacecolor='blue', markersize=7)
    # axs[1].grid(False)
    axs[0].set_xlabel('x')
    axs[0].legend([r'Gaussian 1', 'Gaussian 2'], loc='best')
    
    # iterations = np.arange(0,L2.shape[-1],1)
    
    # axs[1].plot(iterations, L2)
    # axs[1].set_xlabel('iterations')
    # axs[1].set_ylabel('L2 loss')
    
    fig.savefig(frames_dir+'frame_{}.png'.format(i))
    
    # if i == 0 or i == int(cycles*0.25) or i == int(cycles*0.5) or i == cycles-1:
    #     fig.savefig(frames_dir+'Nsensors_{}n_frame_{}_{}.png'.format(nfac+1,i,arange))
    
    
    plt.close()

    
def plot_gif(frames_dir, iterations):
    
    frames = []
    
    for i in range(iterations+1):
        image = imageio.v2.imread(frames_dir+'frame_{}.png'.format(i))
        frames.append(image)
        
    imageio.mimsave(frames_dir+'Gaussian.gif', frames, duration=5, loop = 1)
    
    
# Launching code:
define_measures()
    
    
    
    
    
    
    
    
    
    
