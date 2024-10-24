#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 15:59:55 2023

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
from cbx.scheduler import scheduler, multiply

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

def projObsGraFixSupp(target_field: torch.Tensor,
                    measures: torch.Tensor,
                    field_coordinates: torch.Tensor,
                    prob_name,
                    params_opt = {'optimizer': 'SGD', 'lr': 0.001, 'nmax': 100,'type_loss':'W2', 'gamma':1},
                    # params_sinkhorn_bary = {'blur': 0.001 , 'p':2, 'scaling_N':300,'backward_iterations':0},
                    distance_target_measures = None):
    N = measures.shape[1]
    weights = torch.nn.Parameter(torch.ones((1,N), dtype=dtype, device=device)/N)
    gamma = params_opt['gamma']
    niter = params_opt['nmax']
    
    D = dimension(target_field)
    
    field = field_coordinates.squeeze()
    
    print('type(field)',type(field))
    
    no_of_sensors    = N+1
    sensor_locations = torch.linspace(field[0].item()*0.8, field[-1].item()*0.8, no_of_sensors).tolist()
    # sensor_locations = [0.3, 0.7, 1.0, 1.3, 1.6, 1.8, 2.1]
    # print(sensor_locations)
    sensor_locations = torch.tensor([sensor_locations])
    
    def get_optimizer(weights):
        if params_opt['optimizer'] == 'RMSprop':
            return torch.optim.Adam([weights], lr=params_opt['lr'])
        else:
            return torch.optim.SGD([weights], lr=params_opt['lr'], momentum=0.5)
        
    optimizer = get_optimizer(weights)
    scheduler = ExponentialLR(optimizer, gamma=0.9)
    # scheduler = ExponentialLR(optimizer, gamma=0.1)
    
    S = torch.arange(N,device=device)
    evolution = {'loss':[],'support':[],'weight':[], 'true_error':[]}

    ker, cell_indices = convolution_kernel_v2(field_coordinates, sensor_locations)
    
    # class observables:
    #     def __init__(self, field, field_coordinates, 
    #                   params_oberservables = {'kernel': 'Dirac'}):
    #         self.field = field
    #         self.field_coordinates = field_coordinates
    #         self.params_observables = params_oberservables
    #         self.h = self.field_coordinates[1]-self.field_coordinates[0]
            
    #     def compute_observables(self):
    #         """
    #         Computes observables through a convolution operation, 
    #         based on the choice of a kernel
    #         """
    #         obs_field = torch.zeros(len(self.field_coordinates), device=device)
    #         return torch.matmul(ker,self.field.squeeze()) # in (B,K,N) format
    
    # observation = observables(target_field, field_coordinates)
    # obs_target_field  = observation.compute_observables()
    
    # tmp_obs_target    = obs_target_field[obs_target_field.nonzero()].squeeze()
    # tmp_obs_target    = obs_target_field[indices]
    # tmp_obs_target    = obs_target_field.squeeze()
    # A = T.squeeze()[T.squeeze().nonzero()].squeeze()
    
    tmp_loss = 1e20; tols = 1e-15
    # Initial value for n_keep:
    n_keep = N
    
    frames_dir = check_create_dir(results_dir+'gif_'+prob_name+'/')
    
    for iter in tqdm(range(niter)):
    # for iter in range(niter):
        evolution['weight'].append(weights.detach().clone())
        optimizer.zero_grad()
        if D == 1:
            # print('weights',weights)
            bary = ImagesBarycenter_1d(measures=measures, weights=weights)
            barycenter_observations = torch.matmul(ker, bary.squeeze())
            target_observations     = torch.matmul(ker, target_field.squeeze())
            
            # loss = len(indices)*torch.nn.MSELoss()(tmp_obs_target[None, None, :], tmp_obs_bary[None, None, :])
            # loss = len(indices)*torch.nn.MSELoss()(obs_target_field[None, None, indices]/obs_target_field[indices].sum(), 
            #                                         obs_bary[None, None, indices]/obs_bary[indices].sum())
            
            loss = torch.norm((barycenter_observations/barycenter_observations.sum() \
                               -target_observations/target_observations.sum()), p=2, dim = -1)
        else:
            print('Spatial dimenstion >1D')

        loss.backward()
        optimizer.step()
        # scheduler.step()
        
        #DEBUG:
        # print('loss.grad',loss.grad)
        # print('weights.grad',weights.grad)
        
        #DEBUG
        # print('Wts before', weights)

        # keep n largest value 
        if n_keep < N:
            index_sort= torch.argsort(weights.data,dim=1,descending=True).flatten()
            index_keep = index_sort[:n_keep]
            index_zero = index_sort[n_keep:]
            weights.data[:,index_zero]=0
            
            # projection on the simplex
            weights.data[:,index_keep] = mat2simplex(weights[:,index_keep]*gamma)
            # print('Here!')
        else:
            weights.data= mat2simplex(weights*gamma)
        S_index= torch.nonzero(torch.where(weights > 1.e-4, weights, 0.).flatten()).flatten()
        n_keep = S_index.shape[0]
        #print('Iter {}, weights {}, loss = {}'.format(iter,weights,loss.item()))
    
        #DEBUG:
        # print('Wts after', weights)
        
        #print('supp', S[S_index])
        evolution['loss'].append(loss.item())
        evolution['support'].append( S[S_index].cpu())
        
        
        ######## Plotting frames begins ########
        sensor_placement = torch.zeros(field_coordinates.shape[0])
        
        fig, axs = plt.subplots(1, sharex=True, sharey=True)
        
        sensor_placement = torch.zeros(field_coordinates.shape[0])
        
        axs.plot(field_coordinates.squeeze().cpu().numpy(),target_field.cpu().squeeze().numpy(), 'black')
        
        axs.plot(field_coordinates.squeeze().cpu().numpy(),bary.cpu().detach().squeeze().numpy())
        
        axs.plot(field_coordinates.squeeze().cpu().numpy(), sensor_placement.cpu().numpy(), markevery = cell_indices, ls = "", 
                  marker="o", markerfacecolor='blue', markersize=10)
        
        axs.grid(False)
        # axs.right_axs.grid(False)
        
        axs.set_xlabel('x')
        # axs.set_ylabel('Observables')
        axs.set_ylabel(r'$\mu_{normalized}$', rotation=90)
        axs.legend([r'Target', 'Bary approx'],loc='best')
        
        axs2 = axs.twinx()
        
        for k,l in enumerate(cell_indices): 
            tmp = ker[l][:].cpu().numpy()
            axs2.plot(field_coordinates.squeeze().cpu().numpy(), tmp, 'blue', alpha = 0.2, 
                      linestyle='dotted')
            axs2.fill_between(field_coordinates.squeeze().cpu().numpy(), tmp, 0, color='red', alpha=.05)
                
        axs2.grid(False)
        
        fig.savefig(frames_dir+'frame_{}.png'.format(iter))
        
        plt.close()
        
        ######## Plotting frames ends ########
        
        params_opt['lr'] = params_opt['lr']/2
        
        if tmp_loss - loss.item() > tols:
            tmp_loss = loss
            # print('Modify to add tols to the grad of the loss. Remove print when implemented')
        else:
            print('Reached grad loss under tolerance ({})'.format(tols))
            break
    
    plot_gif(iter, frames_dir, prob_name)
            
    # print('iter', iter, 'loss', loss)
    return target_observations, barycenter_observations, bary, weights, cell_indices, evolution

def plot_gif(iterations, frames_dir, prob_name):
    frames = []
    
    for i in range(iterations+1):
        image = imageio.v2.imread(frames_dir+'frame_{}.png'.format(i))
        frames.append(image)
        
    imageio.mimsave(frames_dir+prob_name+'.gif', frames, duration=5, loop = 1)
    
# def convolution_kernel(field_coordinates):
def convolution_kernel(field_coordinates, sigma):    
    
    field = field_coordinates.squeeze()
    
    npoints = field.shape[-1]
    
    support = torch.linspace(field[0].item(), field[-1].item(), npoints, device = device)
    
    sigma = 2.0 #increase for higher influence from the neighbours
    
    ker = torch.zeros(field.shape[-1],field.shape[-1], dtype = dtype, device = device)
    
    h   = abs(field[0]-field[1]).item()
    
    for i,j in enumerate(field):
        # if i == 220 or i == 300: #Only for Burger
        #     sigma = 0.1
        # else:
        #     sigma = 1.0
        
        ker[i] = h*torch.exp(-0.5*((support-j)/sigma)**2)/((sigma*(2*math.pi)**0.5)+1e-10)
    
    return ker

def optimize_over_wts(target: torch.tensor,
                      wts: torch.tensor,
                           measures: torch.Tensor,
                           field_coordinates: torch.Tensor,
                           sensor_locations: torch.Tensor,
                           # sensor_locations: torch.nn.parameter.Parameter,
                           params_opt = {'optimizer': 'Adam', 'lr': 0.001, 'nmax': 100,'type_loss':'W2', 'gamma':1},
                           params_sinkhorn_bary = {'blur': 0.001 , 'p':2, 'scaling_N':100,'backward_iterations':0}):

    eps = 1e-3; tau = 1.
    Loss = SamplesLoss("sinkhorn", blur = eps, scaling=.9, debias=True)
    
    N = measures.shape[1]
    D = dimension(measures)
    
    field = field_coordinates.squeeze()
    
    no_of_sensors = sensor_locations.shape[-1]
    # no_of_sensors    = 6
    # sensor_locations = [[-1.5, 0.0, 0.5, 1.1, 1.6, 1.9]]
    # sensor_locations = torch.tensor(sensor_locations)

    # x        = torch.linspace(-0.5, 0.5, N)
    # gaussian = lambda x : torch.exp(-0.5*((x)/2)**2)/((2*(2*math.pi)**0.5))
    # g        = gaussian(x)/gaussian(x).sum()
    
    # weights  = torch.nn.Parameter(torch.ones((1,N), dtype=dtype, device=device)/N)
    weights = torch.nn.Parameter(torch.tensor([wts.tolist()], dtype=dtype, device=device))
    
    gamma = params_opt['gamma']
    niter = params_opt['nmax']
    
    def get_optimizer(weights):
        if params_opt['optimizer'] == 'Adam':
            return torch.optim.Adam([weights], lr=params_opt['lr'], maximize=False)
        else:
            return torch.optim.SGD([weights], lr=params_opt['lr'], momentum=0.9, maximize=False)
        
    optimizer = get_optimizer(weights)
    scheduler = ExponentialLR(optimizer, gamma=0.9)
    
    ker, cell_indices = convolution_kernel_v2(field_coordinates, sensor_locations)
    
    S = torch.arange(N,device=device)
    evolution = {'loss':[],'weights':[],'sup_stability_const':[]}
    
    tmp_loss = 1e20; tols = 1e-18

    for iter in tqdm(range(niter)):
        evolution['weights'].append(weights.detach().clone())
        
        optimizer.zero_grad()
        
        bary = ImagesBarycenter_1d(measures=measures, weights=weights)
        
        observation_target = torch.matmul(ker,target.squeeze())
        observation_bary   = torch.matmul(ker,bary.squeeze())
        
        W2 = ImagesLoss(target, bary) #Squared W_2
        
        # W2_Lp = tau*ImagesLoss(bary1, bary2, scaling=0.8) 
        # + (1.-tau)*torch.norm((bary1-bary2), p = 2, dim = -1)
        
        L2 = torch.nn.MSELoss()(observation_target[None, None, :],
                                  observation_bary[None, None, :])
        
        # L2 = torch.norm((observation_target-observation_bary), p=2, dim = -1)
        
        loss = L2
        
        loss.backward(retain_graph=True)
        optimizer.step()
        # scheduler.step()
        
        #DEBUG
        # print('weights1.requires_grad', weights1.requires_grad)
        # print('weights2.requires_grad', weights2.requires_grad)
        
        # print('weights1.is_leaf', weights1.is_leaf)
        # print('weights2.is_leaf', weights2.is_leaf)
        
        # print('sensor_locations.requires_grad', sensor_locations.requires_grad)
        # print('sensor_locations.is_leaf', sensor_locations.is_leaf)
        
        weights.data= mat2simplex(weights*gamma)
        
        # if tmp_loss - loss.item() >= tols:
        #     tmp_loss = loss.item()
        # else:
        #     # print('Reached grad loss under tolerance ({})'.format(tols))
        #     break
        
        evolution['loss'].append(loss.item())
        # evolution['sup_stability_const'].append(1./(loss.item()**0.5))
        evolution['sup_stability_const'].append(loss.item()**0.5)
        
    return loss, evolution, ker, cell_indices, bary, W2, weights.clone().detach().squeeze()
    
def reconstruct_target_w_dynamic_sensors(target: torch.Tensor,
                       measures: torch.Tensor,
                       field_coordinates: torch.Tensor,
                       prob_name,
                       params_opt = {'optimizer': 'Adam', 'lr': 0.001, 'nmax': 100,'type_loss':'W2', 'gamma':1}):
    
    N = measures.shape[1]
    D = dimension(measures)
    
    field = field_coordinates.squeeze()
    
    # no_of_sensors    = N+4
    # sensor_locations = torch.linspace(field[0].item()*0.9, field[-1].item()*0.9, no_of_sensors).tolist()
    # sensor_locations = torch.nn.Parameter(torch.tensor([sensor_locations], dtype=dtype, device=device))
    
    gamma = params_opt['gamma']
    niter = params_opt['nmax']
    
    def get_optimizer(params):
        if params_opt['optimizer'] == 'Adam':
            return torch.optim.Adam([params], lr=params_opt['lr'],
                                    betas=(0.9, 0.999), amsgrad=True, maximize=True)
        elif params_opt['optimizer'] == 'SGD':
            return torch.optim.SGD([params], lr=params_opt['lr'], momentum=0.9, maximize=True)
        elif params_opt['optimizer'] == 'ASGD':
            return torch.optim.ASGD([params], lr=params_opt['lr'], lambd=0.0001,
                                    alpha=0.75, t0=1000000.0, weight_decay=0,
                                    foreach=None, maximize=True, differentiable=False)
        # elif params_opt['optimizer'] == 'Adagrad':
        #     return torch.optim.Adagrad([params], lr=params_opt['lr'],
        #                                lr_decay=0, weight_decay=0,
        #                                initial_accumulator_value=0, eps=1e-10,
        #                                foreach=None, *, maximize=True, differentiable=False)
        
    # optimizer = get_optimizer(sensor_locations)
    # scheduler = ExponentialLR(optimizer, gamma=1.1)    
    
    # weights  = torch.nn.Parameter(torch.ones((1,N), dtype=dtype, device=device)/N)
    # weights = torch.nn.Parameter(torch.tensor([g.tolist()], dtype=dtype, device=device))
        
    S = torch.arange(N,device=device)
    # inf_evolution = {'inf_sup_stability_const':[], 'sup_stability_const':[],'sensor_locations':[], 'iter':[]}
    
    # tmp_loss = 1e20; tols = 1e-18
    
    frames_dir = check_create_dir(results_dir+'Reconstruct_'+prob_name+'/'+'Trial/')
    
    # inf_evolution = {'inf_sup_stability_const':[], 'sup_stability_const':[],'sensor_locations':[], 'iter':[]}
    
    for nfac in range(3):
        
        x        = torch.linspace(-0.5, 0.5, N)
        gaussian = lambda x : torch.exp(-0.5*((x)/2)**2)/((2*(2*math.pi)**0.5))
        wts      = gaussian(x)/gaussian(x).sum()
        
        print('***** m = {}n'.format(nfac+1))
        
        no_of_sensors = (nfac+1)*N
        
        sensor_locations = torch.linspace(field[0].item()*0.9, field[-1].item()*0.9, no_of_sensors).tolist(); arange = 'even'
        # sensor_locations = torch.linspace(-1, 3., no_of_sensors).tolist(); arange = 'central'
        sensor_locations = torch.nn.Parameter(torch.tensor([sensor_locations], dtype=dtype, device=device))
        
        optimizer = get_optimizer(sensor_locations)
        scheduler = ExponentialLR(optimizer, gamma=0.9)
        
        tmp_loss = 1e20; tols = 1e-18
        
        inf_evolution = {'inf_sup_stability_const':[], 'sup_stability_const':[],'sensor_locations':[], 'iter':[]}
        
        Filename = frames_dir+'batch_wts_'+str(no_of_sensors)+'_'+arange

        for iter in tqdm(range(niter)):        
            inf_evolution['sensor_locations'].append(sensor_locations.detach().cpu().clone())
            
            print('sensor_locations.data', sensor_locations.data)
            
            if iter == 0:
                sensors_history = sensor_locations.detach().cpu().clone().reshape(no_of_sensors,1)
            else:
                sensors_history = torch.concatenate((sensors_history,
                                                     sensor_locations.detach().cpu().clone().reshape(no_of_sensors,1)), axis=1)
            optimizer.zero_grad()
            
            sup_loss, sup_evolution, ker, \
            cell_indices, barycenter, W2, wts = optimize_over_wts(target, wts, measures,
                                                              field_coordinates,
                                                              sensor_locations)        
            inf_loss = sup_loss
            
            ######## Writing wts to file    ########
            file = open(Filename+'.txt', "w")
            # for i in inf_evolution['inf_sup_stability_const']: file.write("%s\n" % i)
            for i in wts: file.write("%s\n" % i)
            file.close()
            
            ######## Plotting frames begins ########
            fig, axs = plt.subplots(2, sharex=True, sharey=True)
            
            sensor_placement = torch.zeros(field_coordinates.shape[0])
            # print('inf_evolution: ', inf_evolution['sensor_locations'])
            # indices = inf_evolution['sensor_locations']
            # indices = indices.tolist()
            
            # axs.plot(field.cpu().numpy(), target_field.cpu().squeeze().numpy(), 'black')
            
            # axs.plot(field_coordinates.squeeze().cpu().numpy(),bary.cpu().detach().squeeze().numpy())
            
            for i in range(measures.shape[1]): 
                axs[0].plot(field.cpu().numpy(), torch.squeeze(measures.cpu())[i].numpy())
            
            axs[0].plot(field.cpu().numpy(), sensor_placement.cpu().numpy(), markevery = cell_indices, ls = "", 
                      marker="o", markerfacecolor='blue', markersize=10)
            
            axs[0].grid(False)
            
            axs[0].set_xlabel('x')
            # axs.set_ylabel('Observables')
            # axs.set_ylabel(r'$\mu_{normalized}$', rotation=90)
            # axs[0].set_title(r'$\inf_{x\in\Omega}\, \sup_{\Lambda_1,\Lambda_2} \frac{W_2(\mu,\nu)}{MSE(\mu,\nu)}$')
            axs[0].set_title(r'Reconstructing target. w/ {}ly dist. sensors'.format(arange))
            # axs[0].set_title(r'Computing Lip. const. w/ centrally dist. sensors')
            # axs.legend([r'Target', 'Bary approx'],loc='best')
            
            # axs2 = axs[0].twinx()
            
            # # for k,l in enumerate(cell_indices): 
            # #     tmp = ker[l][:].cpu().detach().numpy()
            # #     axs2.plot(field_coordinates.squeeze().cpu().numpy(), tmp, 'blue', alpha = 0.1, 
            # #               linestyle='dotted')
            # #     axs2.fill_between(field_coordinates.squeeze().cpu().numpy(), tmp, 0, color='blue', alpha=.03)
                
            # for l in range(no_of_sensors): 
            #     tmp = ker[l][:].cpu().detach().numpy()
            #     axs2.plot(field_coordinates.squeeze().cpu().numpy(), tmp, 'blue', alpha = 0.1, 
            #               linestyle='dotted')
            #     axs2.fill_between(field_coordinates.squeeze().cpu().numpy(), tmp, 0, color='blue', alpha=.03)
                    
            # axs2.grid(False)
            
            axs[1].plot(field.cpu().numpy(), target.cpu().detach().squeeze().numpy())
            axs[1].plot(field.cpu().numpy(), barycenter.cpu().detach().squeeze().numpy())
            axs[1].plot(field.cpu().numpy(), sensor_placement.cpu().numpy(), markevery = cell_indices, ls = "", 
                      marker="o", markerfacecolor='blue', markersize=10)
            # axs[1].grid(False)
            axs[1].set_xlabel('x')
            axs[1].legend([r'Target', 'Barycenter'],loc='best')
            # axs[1].title(r'$\inf_{x\in\Omega}\, \sup_{\Lambda_1,\Lambda_2} \frac{W_2(\mu,\nu)}{MSE(\mu,\nu)}$')
            
            axs2 = axs[1].twinx()
            
            # for k,l in enumerate(cell_indices): 
            #     tmp = ker[l][:].cpu().detach().numpy()
            #     axs2.plot(field_coordinates.squeeze().cpu().numpy(), tmp, 'blue', alpha = 0.1, 
            #               linestyle='dotted')
            #     axs2.fill_between(field_coordinates.squeeze().cpu().numpy(), tmp, 0, color='blue', alpha=.03)
                
            for l in range(no_of_sensors): 
                tmp = ker[l][:].cpu().detach().numpy()
                axs2.plot(field_coordinates.squeeze().cpu().numpy(), tmp, 'blue', alpha = 0.1, 
                          linestyle='dotted')
                axs2.fill_between(field_coordinates.squeeze().cpu().numpy(), tmp, 0, color='blue', alpha=.03)
                    
            axs2.grid(False)
            
            fig.savefig(frames_dir+'frame_{}.png'.format(iter))
            
            if iter == 0 or iter == 50 or iter == 100 or iter == niter-1:
                fig.savefig(frames_dir+'Nsensors_{}n_frame_{}_{}.png'.format(nfac+1,iter,arange))
            
            # plt.title(r"Hello \Tex", fontsize=6)
            plt.close()        
                
            inf_loss.backward()
            optimizer.step()
            # scheduler.step()
            
            #DEBUG
            # print('sensor_locations.requires_grad', sensor_locations.requires_grad)
            # print('sensor_locations.is_leaf', sensor_locations.is_leaf)
            
            # sensor_locations.data.clamp_(field[0].item(), field[-1].item())
            
            # if tmp_loss - inf_loss.item() >= tols:
            #     tmp_loss = inf_loss.item()
            # else:
            #     # print('Reached grad loss under tolerance ({})'.format(tols))
            #     break
            
            inf_evolution['iter'].append(iter)
            inf_evolution['inf_sup_stability_const'].append(inf_loss.item())
            inf_evolution['sup_stability_const'].append(sup_evolution['sup_stability_const'][-1])

        #DEBUG
        print('sensors_history.shape', sensors_history.shape)
        df = pd.DataFrame(sensors_history.numpy())
        df.to_csv(frames_dir+'sensor_history_'+arange+'_'+str(no_of_sensors), index=False)
        print('Sensor locations written to file.')
        
        plot_sensor_trajectory(iter, no_of_sensors, sensors_history, frames_dir,
                                   arange, niter, prob_name)
        
        prob_name_gif = str('batch_')+prob_name+str('_{}_{}'.format(no_of_sensors,arange))
        
        plot_gif(iter, frames_dir, prob_name_gif)
            
        Filename = frames_dir+'batch_C_L_'+str(no_of_sensors)+'_'+arange
        file = open(Filename+'.txt', "w")
        # for i in inf_evolution['inf_sup_stability_const']: file.write("%s\n" % i)
        for i in inf_evolution['sup_stability_const']: file.write("%s\n" % i)
        file.close()
        print('Writing inf_loss to file complete')
        
        if nfac == 0:
            C_s = np.array([inf_evolution['sup_stability_const']])
        else:
            tmp = np.array([inf_evolution['sup_stability_const']])
            C_s = np.concatenate((C_s, tmp), axis=0)
            print('shape C_s', C_s.shape)
            
    plot_graphs(niter, nfac, C_s, frames_dir, prob_name)
    
    return inf_loss, inf_evolution, sup_evolution

def reconstruct_target_w_static_sensors(target: torch.Tensor,
                       measures: torch.Tensor,
                       field_coordinates: torch.Tensor,
                       prob_name, saved_dir, arange, tstep,
                       params_opt = {'optimizer': 'Adam', 'lr': 0.01, 'nmax': 100,'type_loss':'W2', 'gamma':1}):
    
    # frames_dir = check_create_dir(saved_dir+'/'+'reconstruct_target/')
    frames_dir = check_create_dir(saved_dir+'/'+'reconstruct_travelling_target_trial/{}/t{}/'.format(arange,tstep))
    
    N = measures.shape[1]
    D = dimension(measures)
    
    field = field_coordinates.squeeze()
    
    # no_of_sensors    = N+4
    # sensor_locations = torch.linspace(field[0].item()*0.9, field[-1].item()*0.9, no_of_sensors).tolist()
    # sensor_locations = torch.nn.Parameter(torch.tensor([sensor_locations], dtype=dtype, device=device))
    
    # inf_evolution = {'inf_sup_stability_const':[], 'sup_stability_const':[],'sensor_locations':[], 'iter':[]}
    
    # tmp_loss = 1e20; tols = 1e-18    
    
    # inf_evolution = {'inf_sup_stability_const':[], 'sup_stability_const':[],'sensor_locations':[], 'iter':[]}
    
    for nfac in range(3):
        
        print('***** m = {}n'.format(nfac+1))
        
        no_of_sensors = (nfac+1)*N
        
        # sensor_locations = torch.linspace(field[0].item()*0.9, field[-1].item()*0.9, no_of_sensors).tolist(); arange = 'even'
        sensor_locations = torch.load(saved_dir+'{}_{}_sensor-coordinates'.format(no_of_sensors,arange))
        sensor_locations = sensor_locations.to(device)
        
        x        = torch.linspace(-0.5, 0.5, N)
        gaussian = lambda x : torch.exp(-0.5*((x)/2)**2)/((2*(2*math.pi)**0.5))
        wts      = gaussian(x)/gaussian(x).sum()
        weights  = torch.nn.Parameter(torch.tensor([wts.tolist()], dtype=dtype, device=device))
        
        gamma = params_opt['gamma']
        niter = params_opt['nmax']
        
        def get_optimizer(weights):
            if params_opt['optimizer'] == 'Adam':
                return torch.optim.Adam([weights], lr=params_opt['lr'], maximize=False)
            else:
                return torch.optim.SGD([weights], lr=params_opt['lr'], momentum=0.9, maximize=False)
            
        optimizer = get_optimizer(weights)
        scheduler = ExponentialLR(optimizer, gamma=0.95)
        
        ker, cell_indices = convolution_kernel_v2(field_coordinates, sensor_locations)
        
        S = torch.arange(N,device=device)
        evolution = {'loss':[], 'W2_loss':[],'weights':[],'C_s':[]}
        
        Filename = frames_dir+'batch_wts_'+str(no_of_sensors)

        for iter in tqdm(range(niter)):
            evolution['weights'].append(weights.detach().clone())
            
            optimizer.zero_grad()
            
            bary = ImagesBarycenter_1d(measures=measures, weights=weights)
            
            observation_target = torch.matmul(ker,target.squeeze())
            observation_bary   = torch.matmul(ker,bary.squeeze())
            
            W2 = ImagesLoss(target, bary) #Squared W_2
            
            # W2_Lp = tau*ImagesLoss(bary1, bary2, scaling=0.8) 
            # + (1.-tau)*torch.norm((bary1-bary2), p = 2, dim = -1)
            
            L2 = torch.nn.MSELoss()(observation_target[None, None, :],
                                      observation_bary[None, None, :])
            
            # L2 = torch.norm((observation_target-observation_bary), p=2, dim = -1)
            
            loss = L2
            
            loss.backward(retain_graph=True)
            optimizer.step()
            # scheduler.step()
            
            weights.data= mat2simplex(weights*gamma)
            
            # if tmp_loss - loss.item() >= tols:
            #     tmp_loss = loss.item()
            # else:
            #     # print('Reached grad loss under tolerance ({})'.format(tols))
            #     break
                        
            ######## Plotting frames begins ########
            fig, axs = plt.subplots(2, sharex=True, sharey=True)
            
            sensor_placement = torch.zeros(field_coordinates.shape[0])
            # print('inf_evolution: ', inf_evolution['sensor_locations'])
            # indices = inf_evolution['sensor_locations']
            # indices = indices.tolist()
            
            # axs.plot(field.cpu().numpy(), target_field.cpu().squeeze().numpy(), 'black')
            
            # axs.plot(field_coordinates.squeeze().cpu().numpy(),bary.cpu().detach().squeeze().numpy())
            
            for i in range(measures.shape[1]): 
                axs[0].plot(field.cpu().numpy(), torch.squeeze(measures.cpu())[i].numpy())
            
            axs[0].plot(field.cpu().numpy(), sensor_placement.cpu().numpy(), markevery = cell_indices, ls = "", 
                      marker="o", markerfacecolor='blue', markersize=10)
            
            axs[0].grid(False)
            
            axs[0].set_xlabel('x')
            # axs.set_ylabel('Observables')
            # axs.set_ylabel(r'$\mu_{normalized}$', rotation=90)
            # axs[0].set_title(r'$\inf_{x\in\Omega}\, \sup_{\Lambda_1,\Lambda_2} \frac{W_2(\mu,\nu)}{MSE(\mu,\nu)}$')
            axs[0].set_title(r'Reconstructing target using observables')
            
            axs[1].plot(field.cpu().numpy(), target.cpu().detach().squeeze().numpy())
            axs[1].plot(field.cpu().numpy(), bary.cpu().detach().squeeze().numpy())
            axs[1].plot(field.cpu().numpy(), sensor_placement.cpu().numpy(), markevery = cell_indices, ls = "", 
                      marker="o", markerfacecolor='blue', markersize=10)
            # axs[1].grid(False)
            axs[1].set_xlabel('x')
            axs[1].legend([r'Target', 'Barycenter'],loc='best')
            # axs[1].title(r'$\inf_{x\in\Omega}\, \sup_{\Lambda_1,\Lambda_2} \frac{W_2(\mu,\nu)}{MSE(\mu,\nu)}$')
            
            axs2 = axs[1].twinx()
            
            for l in range(no_of_sensors): 
                tmp = ker[l][:].cpu().detach().numpy()
                axs2.plot(field_coordinates.squeeze().cpu().numpy(), tmp, 'blue', alpha = 0.1, 
                          linestyle='dotted')
                axs2.fill_between(field_coordinates.squeeze().cpu().numpy(), tmp, 0, color='blue', alpha=.03)
                    
            axs2.grid(False)
            
            fig.savefig(frames_dir+'frame_{}.png'.format(iter))
            
            if iter == 0 or iter == int(niter*0.25) or iter == int(niter*0.5) or iter == niter-1:
                fig.savefig(frames_dir+'Nsensors_{}n_frame_{}_{}.png'.format(nfac+1,iter,arange))
            
            # plt.title(r"Hello \Tex", fontsize=6)
            plt.close()        
            
            evolution['loss'].append(loss.item())
            evolution['W2_loss'].append(W2.item())
            evolution['weights'].append(weights.detach().squeeze().clone())
            evolution['C_s'].append(W2/L2)

        
        prob_name_gif = str('Reconstruct_')+prob_name+str('_{}_{}_sensors'.format(no_of_sensors,arange))
        
        plot_gif(iter, frames_dir, prob_name_gif)
            
        Filename = frames_dir+'C_s_reconstruct_'+prob_name+str('_{}_{}_sensors'.format(no_of_sensors,arange))
        file = open(Filename+'.txt', "w")
        # for i in inf_evolution['inf_sup_stability_const']: file.write("%s\n" % i)
        for i in evolution['loss']: file.write("%s\n" % i)
        file.close()
        print('Writing C_s to file complete')
        
        Filename = frames_dir+'W2_reconstruct_'+prob_name+str('_{}_{}_sensors'.format(no_of_sensors,arange))
        file = open(Filename+'.txt', "w")
        # for i in inf_evolution['inf_sup_stability_const']: file.write("%s\n" % i)
        for i in evolution['W2_loss']: file.write("%s\n" % i)
        file.close()
        print('Writing W2 to file complete')
        
        if nfac == 0:
            C_s     = np.array([ evolution['loss']])
            W2_loss = np.array([ evolution['W2_loss']])
        else:
            tmp  = np.array([ evolution['loss']])
            tmp2 = np.array([ evolution['W2_loss']])
            C_s     = np.concatenate((C_s, tmp), axis=0)
            W2_loss = np.concatenate((W2_loss, tmp2), axis=0)
            print('shape C_s', C_s.shape)
            
    plot_graphs(niter, nfac, C_s, frames_dir, prob_name+'_C_s_', arange)
    plot_graphs(niter, nfac, W2_loss, frames_dir, prob_name+'_W2_loss_', arange)

def convolution_kernel_v2(field_coordinates, sensor_locations):

    #DEBUG
    # print('sensor_locations', sensor_locations.shape)
    # print('field_coordinates', field_coordinates.shape)
    
    # loc   = sensor_locations.clone().detach()
    
    m = sensor_locations.shape[-1]
    
    field = field_coordinates.squeeze() #; print(field.shape[-1])
    
    npoints = field.shape[-1]
    
    h   = abs(field[0].item()-field[1].item())
    
    cell_indices = []
    field_as_list            = field.tolist()
    sensor_locations_as_list = sensor_locations.tolist()[0]
    #DEBUG:
    # print('dtype field', type(field), field.shape)  
    
    # print('dtype field_as_list', type(field_as_list))
    
    # print('field_as_list',field_as_list)
    
    #DEBUG
    # print('sensor_locations_as_list', sensor_locations_as_list)
    # print('h', h)
            
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
        
    # cell_indices = adjust_sensor_location(cell_indices)
    
    print('cell_indices',cell_indices)
    # sigma = 0.05 #increase for higher influence from the neighbours
    sigma = (field[-1]-field[0])*0.015
    
    # ker = torch.zeros(m,field.shape[-1], dtype = dtype, device = device)
    
    f   = lambda x : (x>=0.)*1
    g   = lambda x : f(x+0.25)-f(x-0.25)
    
    field = field.repeat(m,1)
    
    # DEBUG:
    # print('field.shape', field.shape)
    # print('sensor_locations', sensor_locations.shape)
        
    gaussian = lambda x : (torch.exp(-0.5*((field-x)/sigma)**2)/((sigma*(2*math.pi)**0.5)))    
    
    ker = gaussian(sensor_locations.T)
    
    return ker, cell_indices
    

def sup_stability_const(measures: torch.Tensor, 
                        field_coordinates: torch.Tensor, 
                        sensor_locations: torch.nn.parameter.Parameter,
                        wts: torch.Tensor,
                        params_opt = {'optimizer': 'Adam', 'lr': 0.01, 'nmax': 50,'type_loss':'W2', 'gamma':1}):

    # eps = 1e-3; tau = 1.
    # Loss = SamplesLoss("sinkhorn", blur = eps, scaling=.9, debias=True)
    
    eps = 0.e-8
    
    N = measures.shape[1]
    D = dimension(measures)
    
    field = field_coordinates.squeeze()
    
    no_of_sensors = sensor_locations.shape[-1]; print('No. of sensors', no_of_sensors)
    # no_of_sensors    = 6
    # sensor_locations = [[-1.5, 0.0, 0.5, 1.1, 1.6, 1.9]]
    # sensor_locations = torch.tensor(sensor_locations)

    x        = torch.linspace(-0.5, 0.5, N)
    gaussian = lambda x : torch.exp(-0.5*((x)/2)**2)/((2*(2*math.pi)**0.5))
    g        = gaussian(x)/gaussian(x).sum()
    
    weights1 = torch.nn.Parameter(torch.ones((1,N), dtype=dtype, device=device)/N)
    # weights2 = torch.nn.Parameter(torch.tensor([[0.5,0,0.5]], dtype=dtype, device=device))
    weights2 = torch.nn.Parameter(torch.tensor([g.tolist()], dtype=dtype, device=device))
    
    gamma = params_opt['gamma']
    niter = params_opt['nmax']
    
    def get_optimizer(params):
        if params_opt['optimizer'] == 'Adam':
            return torch.optim.Adam(params, lr=params_opt['lr'], maximize=True)
        else:
            return torch.optim.SGD(params, lr=params_opt['lr'], momentum=0.9, maximize=False)
    
    # params = [weights1, weights2]
    params = wts
    optimizer = get_optimizer(params)
    scheduler = ExponentialLR(optimizer, gamma=0.99)
    
    ker, cell_indices = convolution_kernel_v2(field_coordinates, sensor_locations)
       
    S = torch.arange(N,device=device)
    evolution = {'loss':[],'weights1':[],'weights2':[], 'sup_stability_const':[]}
    
    tmp_loss = 1e20; tols = 1e-18

    for iter in tqdm(range(niter)):
        evolution['weights1'].append(params[0].detach().clone())
        evolution['weights2'].append(params[1].detach().clone())
        
        optimizer.zero_grad()
        
        bary1 = ImagesBarycenter_1d(measures=measures, weights=params[0])
        bary2 = ImagesBarycenter_1d(measures=measures, weights=params[1])
        
        observation_bary1 = torch.matmul(ker,bary1.squeeze())
        observation_bary2 = torch.matmul(ker,bary2.squeeze())
        
        # observation_bary1 = torch.matmul(bary1.squeeze(),ker)
        # observation_bary2 = torch.matmul(bary2.squeeze(),ker)
        
        W2   = ImagesLoss(bary1, bary2) #Squared W_2
        
        # W2_Lp = tau*ImagesLoss(bary1, bary2, scaling=0.8) 
        # + (1.-tau)*torch.norm((bary1-bary2), p = 2, dim = -1)
        
        L2 = (torch.nn.MSELoss()(observation_bary1[None, None, :],
                                 observation_bary2[None, None, :]))
        
        # Lp = torch.norm((observation_bary1-observation_bary2), p=2, dim = -1)
        
        # loss = torch.pow((Lp/W2), 0.5)
        # loss = torch.pow((W2/Lp), 0.5)
        loss = W2/(L2+eps)
        # loss = L2/W2
        
        loss.backward(retain_graph=True)
        # loss.backward()
        optimizer.step()
        scheduler.step()
        
        #DEBUG
        # print('weights1.requires_grad', weights1.requires_grad)
        # print('weights2.requires_grad', weights2.requires_grad)
        
        # print('weights1.is_leaf', weights1.is_leaf)
        # print('weights2.is_leaf', weights2.is_leaf)
        
        # print('sensor_locations.requires_grad', sensor_locations.requires_grad)
        # print('sensor_locations.is_leaf', sensor_locations.is_leaf)
        
        params[0].data= mat2simplex(params[0]*gamma)
        params[1].data= mat2simplex(params[1]*gamma)
        
        # if tmp_loss - loss.item() >= tols:
        #     tmp_loss = loss.item()
        # else:
        #     # print('Reached grad loss under tolerance ({})'.format(tols))
        #     break
        
        evolution['loss'].append(loss.item())
        # evolution['sup_stability_const'].append(1./(loss.item()**0.5))
        evolution['sup_stability_const'].append(loss.item()**0.5)
        
    return loss, evolution, ker, cell_indices, bary1, bary2, params, L2

def inf_stability_const(target: torch.Tensor, measures: torch.Tensor, 
                        field_coordinates: torch.Tensor,
                        prob_name,
                        params_opt = {'optimizer': 'Adam', 'lr': 0.01, 'nmax': 50,'type_loss':'W2', 'gamma':1}):
    
    N = measures.shape[1]
    D = dimension(measures)
    
    field = field_coordinates.squeeze()
    
    # no_of_sensors    = N+4
    # sensor_locations = torch.linspace(field[0].item()*0.9, field[-1].item()*0.9, no_of_sensors).tolist()
    # sensor_locations = torch.nn.Parameter(torch.tensor([sensor_locations], dtype=dtype, device=device))
    
    gamma = params_opt['gamma']
    niter = params_opt['nmax']
    
    def get_optimizer(params):
        if params_opt['optimizer'] == 'Adam':
            return torch.optim.Adam([params], lr=params_opt['lr'],
                                    betas=(0.9, 0.999), amsgrad=True, maximize=False)
        elif params_opt['optimizer'] == 'SGD':
            return torch.optim.SGD([params], lr=params_opt['lr'], momentum=0.9, maximize=True)
        elif params_opt['optimizer'] == 'ASGD':
            return torch.optim.ASGD([params], lr=params_opt['lr'], lambd=0.0001,
                                    alpha=0.75, t0=1000000.0, weight_decay=0,
                                    foreach=None, maximize=True, differentiable=False)
        # elif params_opt['optimizer'] == 'Adagrad':
        #     return torch.optim.Adagrad([params], lr=params_opt['lr'],
        #                                lr_decay=0, weight_decay=0,
        #                                initial_accumulator_value=0, eps=1e-10,
        #                                foreach=None, *, maximize=True, differentiable=False)
        
    # optimizer = get_optimizer(sensor_locations)
    # scheduler = ExponentialLR(optimizer, gamma=1.1)
    
    S = torch.arange(N,device=device)
    # inf_evolution = {'inf_sup_stability_const':[], 'sup_stability_const':[],'sensor_locations':[], 'iter':[]}
    
    # tmp_loss = 1e20; tols = 1e-18
    
    # frames_dir = check_create_dir(results_dir+'Frames_compute_mu_SG'+prob_name+'/')
    # frames_dir = check_create_dir(results_dir+'Compute_C_s_SG_sigma1'+prob_name+'/')
    # frames_dir = check_create_dir(results_dir+'Compute_C_s_SG_translated_meas_'+prob_name+'/')
    # frames_dir = check_create_dir(results_dir+'Compute_Cs_torch'+prob_name+'/')
    # frames_dir = check_create_dir(results_dir+'Compute_C_s_two_measures/SGD/')
    frames_dir = check_create_dir(results_dir+'Compute_C_s_'+prob_name+'/'+'five_measures/SGD/')
    
    # inf_evolution = {'inf_sup_stability_const':[], 'sup_stability_const':[],'sensor_locations':[], 'iter':[]}
    
    for nfac in range(3):
        
        x        = torch.linspace(-0.5, 0.5, N)
        gaussian = lambda x : torch.exp(-0.5*((x)/2)**2)/((2*(2*math.pi)**0.5))
        g        = gaussian(x)/gaussian(x).sum()
    
        weights1 = torch.nn.Parameter(torch.ones((1,N), dtype=dtype, device=device)/N)
        # weights2 = torch.nn.Parameter(torch.tensor([[0.5,0,0.5]], dtype=dtype, device=device))
        weights2 = torch.nn.Parameter(torch.tensor([g.tolist()], dtype=dtype, device=device))
        
        wts      = [weights1, weights2]
        
        print('***** m = {}n'.format(nfac+1))
        
        no_of_sensors = (nfac+1)*N
        
        sensor_locations = torch.linspace(field[0].item()*0.9, field[-1].item()*0.9, no_of_sensors).tolist(); arange = 'even'
        # sensor_locations = torch.linspace(0., 1., no_of_sensors).tolist(); arange = 'central'
        sensor_locations = torch.nn.Parameter(torch.tensor([sensor_locations], dtype=dtype, device=device))
        
        optimizer = get_optimizer(sensor_locations)
        scheduler = ExponentialLR(optimizer, gamma=0.9)
        
        tmp_loss = 1e20; tols = 1e-18
        
        inf_evolution = {'inf_sup_stability_const':[], 'sup_stability_const':[],'sensor_locations':[], 'iter':[]}
        
        Filename = frames_dir+'batch_wts_'+str(no_of_sensors)+'_'+arange

        for iter in tqdm(range(niter)):        
            inf_evolution['sensor_locations'].append(sensor_locations.detach().cpu().clone())
            
            print('sensor_locations.data', sensor_locations.data)
            
            if iter == 0:
                sensors_history = sensor_locations.detach().cpu().clone().reshape(no_of_sensors,1)
            else:
                sensors_history = torch.concatenate((sensors_history,
                                                     sensor_locations.detach().cpu().clone().reshape(no_of_sensors,1)), axis=1)
            optimizer.zero_grad()
            
            sup_loss, sup_evolution, ker, cell_indices, bary1, bary2, wts, inf_loss = sup_stability_const(measures, field_coordinates, sensor_locations, wts)        
            
            # inf_loss = sup_loss
            
            ######## Writing wts to file    ########
            file = open(Filename+'.txt', "w")
            # for i in inf_evolution['inf_sup_stability_const']: file.write("%s\n" % i)
            for i in wts: file.write("%s\n" % i)
            file.close()
            
            ######## Plotting frames begins ########
            fig, axs = plt.subplots(2, sharex=True, sharey=True)
            
            sensor_placement = torch.zeros(field_coordinates.shape[0])
            # print('inf_evolution: ', inf_evolution['sensor_locations'])
            # indices = inf_evolution['sensor_locations']
            # indices = indices.tolist()
            
            # axs.plot(field.cpu().numpy(), target_field.cpu().squeeze().numpy(), 'black')
            
            # axs.plot(field_coordinates.squeeze().cpu().numpy(),bary.cpu().detach().squeeze().numpy())
            
            for i in range(measures.shape[1]): 
                axs[0].plot(field.cpu().numpy(), torch.squeeze(measures.cpu())[i].numpy())
            
            axs[0].plot(field.cpu().numpy(), sensor_placement.cpu().numpy(), markevery = cell_indices, ls = "", 
                      marker="o", markerfacecolor='blue', markersize=10)
            
            axs[0].grid(False)
            
            axs[0].set_xlabel('x')
            # axs.set_ylabel('Observables')
            # axs.set_ylabel(r'$\mu_{normalized}$', rotation=90)
            # axs[0].set_title(r'$\inf_{x\in\Omega}\, \sup_{\Lambda_1,\Lambda_2} \frac{W_2(\mu,\nu)}{MSE(\mu,\nu)}$')
            axs[0].set_title(r'Computing stability const. w/ {}ly dist. sensors'.format(arange))
            # axs[0].set_title(r'Computing Lip. const. w/ centrally dist. sensors')
            # axs.legend([r'Target', 'Bary approx'],loc='best')
            
            # axs2 = axs[0].twinx()
            
            # # for k,l in enumerate(cell_indices): 
            # #     tmp = ker[l][:].cpu().detach().numpy()
            # #     axs2.plot(field_coordinates.squeeze().cpu().numpy(), tmp, 'blue', alpha = 0.1, 
            # #               linestyle='dotted')
            # #     axs2.fill_between(field_coordinates.squeeze().cpu().numpy(), tmp, 0, color='blue', alpha=.03)
                
            # for l in range(no_of_sensors): 
            #     tmp = ker[l][:].cpu().detach().numpy()
            #     axs2.plot(field_coordinates.squeeze().cpu().numpy(), tmp, 'blue', alpha = 0.1, 
            #               linestyle='dotted')
            #     axs2.fill_between(field_coordinates.squeeze().cpu().numpy(), tmp, 0, color='blue', alpha=.03)
                    
            # axs2.grid(False)
            
            axs[1].plot(field.cpu().numpy(), bary1.cpu().detach().squeeze().numpy())
            axs[1].plot(field.cpu().numpy(), bary2.cpu().detach().squeeze().numpy())
            axs[1].plot(field.cpu().numpy(), sensor_placement.cpu().numpy(), markevery = cell_indices, ls = "", 
                      marker="o", markerfacecolor='blue', markersize=10)
            # axs[1].grid(False)
            axs[1].set_xlabel('x')
            axs[1].legend([r'Barycenter 1', 'Barycenter 2'],loc='best')
            # axs[1].title(r'$\inf_{x\in\Omega}\, \sup_{\Lambda_1,\Lambda_2} \frac{W_2(\mu,\nu)}{MSE(\mu,\nu)}$')
            
            axs2 = axs[1].twinx()
            
            # for k,l in enumerate(cell_indices): 
            #     tmp = ker[l][:].cpu().detach().numpy()
            #     axs2.plot(field_coordinates.squeeze().cpu().numpy(), tmp, 'blue', alpha = 0.1, 
            #               linestyle='dotted')
            #     axs2.fill_between(field_coordinates.squeeze().cpu().numpy(), tmp, 0, color='blue', alpha=.03)
                
            for l in range(no_of_sensors): 
                tmp = ker[l][:].cpu().detach().numpy()
                axs2.plot(field_coordinates.squeeze().cpu().numpy(), tmp, 'blue', alpha = 0.1, 
                          linestyle='dotted')
                axs2.fill_between(field_coordinates.squeeze().cpu().numpy(), tmp, 0, color='blue', alpha=.03)
                    
            axs2.grid(False)
            
            fig.savefig(frames_dir+'frame_{}.png'.format(iter))
            
            if iter == 0 or iter == int(niter*0.25) or iter == int(niter*0.5) or iter == niter-1:
                fig.savefig(frames_dir+'Nsensors_{}n_frame_{}_{}.png'.format(nfac+1,iter,arange))
            
            # plt.title(r"Hello \Tex", fontsize=6)
            plt.close()        
                
            inf_loss.backward()
            optimizer.step()
            # scheduler.step()
            
            # if tmp_loss - inf_loss.item() >= tols:
            #     tmp_loss = inf_loss.item()
            # else:
            #     # print('Reached grad loss under tolerance ({})'.format(tols))
            #     break
            
            inf_evolution['iter'].append(iter)
            inf_evolution['inf_sup_stability_const'].append(inf_loss.item())
            inf_evolution['sup_stability_const'].append(sup_evolution['sup_stability_const'][-1])

        #DEBUG
        print('sensors_history.shape', sensors_history.shape)
        df = pd.DataFrame(sensors_history.numpy())
        df.to_csv(frames_dir+'sensor_history_'+arange+'_'+str(no_of_sensors), index=False)
        print('Sensor locations written to file.')
        
        plot_sensor_trajectory(iter, no_of_sensors, sensors_history, frames_dir,
                                   arange, niter, prob_name)
        
        prob_name_gif = str('batch_')+prob_name+str('_{}_{}'.format(no_of_sensors,arange))
        
        plot_gif(iter, frames_dir, prob_name_gif)
            
        Filename = frames_dir+'batch_C_L_'+str(no_of_sensors)+'_'+arange
        file = open(Filename+'.txt', "w")
        # for i in inf_evolution['inf_sup_stability_const']: file.write("%s\n" % i)
        for i in inf_evolution['sup_stability_const']: file.write("%s\n" % i)
        file.close()
        print('Writing inf_loss to file complete')
        
        if nfac == 0:
            C_s = np.array([inf_evolution['sup_stability_const']])
        else:
            tmp = np.array([inf_evolution['sup_stability_const']])
            C_s = np.concatenate((C_s, tmp), axis=0)
            print('shape C_s', C_s.shape)
        
        torch.save(sensor_locations.detach().cpu().clone(), frames_dir+'{}_{}_sensor-coordinates'.format(no_of_sensors,arange))    
        # x = torch.load(frames_dir+'{}_{}_sensor-coordinates'.format(no_of_sensors,arange))
        # print(x)
        
    plot_graphs(niter, nfac, C_s, frames_dir, prob_name, arange)
    
    # reconstruct_target_w_static_sensors(target, measures,
    #                                     field_coordinates, prob_name, frames_dir, arange)
    
    return inf_loss, inf_evolution, sup_evolution

def plot_graphs(niter, nfac, C_s, frames_dir, prob_name, arange):
    
    fig, axs = plt.subplots(1, sharex=True, sharey=True)

    x = np.arange(0, niter)
    
    # for i in range(nfac): axs.plot(x, C_s[i])
    axs.plot(x, C_s[0])
    axs.plot(x, C_s[1])
    axs.plot(x, C_s[2])
    axs.set_xlabel('niter')
    # axs.set_ylabel(r'$C_L$', rotation=90)
    # axs.set_ylabel(r'$C_S$', rotation=90)
    axs.legend([r'm=n', 'm=2n', 'm=3n'],loc='best')
    plt.yscale("log")
    fig.savefig(frames_dir+prob_name+arange)
    
def plot_sensor_trajectory(iter, no_of_sensors, sensors_history, frames_dir,
                           arange, niter, prob_name):
    
    fig, axs = plt.subplots(1, sharex=True, sharey=True)

    x = np.arange(0, iter+1)
    
    for i in range(no_of_sensors): axs.plot(x, sensors_history[i])

    # axs.legend([r'sensor {}'.format(i+1) for i in range(no_of_sensors)],loc='best')
    axs.set_xlabel('niter')
    axs.set_ylabel(r'$x$', rotation=90)

    fig.savefig(frames_dir+'sensor_history_'+arange+'_'+str(no_of_sensors)+'.eps', format='eps')

def f(measures: torch.Tensor, field_coordinates: torch.Tensor,
      it, frames_dir, prob_name,
      x: np.ndarray):
    #DEBUG:
    print('dtype x',type(x))
    
    params_opt = {'optimizer': 'Adam', 'lr': 0.001, 'nmax': 1,'type_loss':'W2', 'gamma':1}

    eps = 1e-3; tau = 1.
    Loss = SamplesLoss("sinkhorn", blur = eps, scaling=.9, debias=True)
    
    N = measures.shape[1]
    D = dimension(measures)
    
    field = field_coordinates.squeeze()
    
    sensor_locations = torch.tensor(x.squeeze()[None, :])
    
    no_of_sensors = sensor_locations.shape[-1]; print('No. of sensors', no_of_sensors)

    tmp        = torch.linspace(-0.5, 0.5, N)
    gaussian = lambda tmp : torch.exp(-0.5*((tmp)/2)**2)/((2*(2*math.pi)**0.5))
    g        = gaussian(tmp)/gaussian(tmp).sum()
    
    weights1 = torch.nn.Parameter(torch.ones((1,N), dtype=dtype, device=device)/N)
    # weights2 = torch.nn.Parameter(torch.tensor([[0.5,0,0.5]], dtype=dtype, device=device))
    weights2 = torch.nn.Parameter(torch.tensor([g.tolist()], dtype=dtype, device=device))
    
    gamma = params_opt['gamma']
    niter = params_opt['nmax']
    
    def get_optimizer(params):
        if params_opt['optimizer'] == 'Adam':
            return torch.optim.Adam(params, lr=params_opt['lr'], maximize=True)
        else:
            return torch.optim.SGD(params, lr=params_opt['lr'], momentum=0., maximize=True)
    
    params = [weights1, weights2]
    optimizer = get_optimizer(params)
    scheduler = ExponentialLR(optimizer, gamma=0.9)
    
    ker, cell_indices = convolution_kernel_v2(field_coordinates, sensor_locations)
    
    # class observables:
    #     def __init__(self, field, field_coordinates, ker, 
    #                   params_oberservables = {'kernel': 'Dirac'}):
    #         self.field = field
    #         self.field_coordinates = field_coordinates
    #         self.params_observables = params_oberservables
    #         self.h = self.field_coordinates[1]-self.field_coordinates[0]
            
    #     def compute_observables(self):
    #         """
    #         Computes observables through a convolution operation, 
    #         based on the choice of a kernel
    #         """
    #         # obs_field = torch.zeros(len(self.field_coordinates), device=device)
            
    #         return torch.matmul(ker,self.field.squeeze()) # in (B,K,N) format    
    
    S = torch.arange(N,device=device)
    evolution = {'loss':[],'weights1':[],'weights2':[], 'sup_stability_const':[]}
    
    tmp_loss = 1e20; tols = 1e-18

    for iter in tqdm(range(niter)):
        evolution['weights1'].append(params[0].detach().clone())
        evolution['weights2'].append(params[1].detach().clone())
        
        optimizer.zero_grad()
        
        bary1 = ImagesBarycenter_1d(measures=measures, weights=params[0])
        bary2 = ImagesBarycenter_1d(measures=measures, weights=params[1])
        
        observation_bary1 = torch.matmul(ker,bary1.squeeze())
        observation_bary2 = torch.matmul(ker,bary2.squeeze())
        
        W2   = ImagesLoss(bary1, bary2, scaling=0.8) #Check if it is squared W_2
        
        # W2_Lp = tau*ImagesLoss(bary1, bary2, scaling=0.8) 
        # + (1.-tau)*torch.norm((bary1-bary2), p = 2, dim = -1)
        
        # L2 = no_of_sensors*(torch.nn.MSELoss()(observation_bary1[None, None, :]/
        #                            observation_bary1.sum(),observation_bary2[None, None, :]/
        #                            observation_bary2.sum())) #Check if MSE is squared norm
        
        Lp = torch.norm((observation_bary1-observation_bary2), p=2, dim = -1)
        
        # loss = torch.pow((Lp/W2), 0.5)
        # loss = torch.pow((W2/Lp), 0.5)
        # loss = Lp/W2
        loss = W2/Lp
        
        loss.backward()
        optimizer.step()
        # scheduler.step()
        
        #DEBUG
        # print('weights1.requires_grad', weights1.requires_grad)
        # print('weights2.requires_grad', weights2.requires_grad)
        
        # print('weights1.is_leaf', weights1.is_leaf)
        # print('weights2.is_leaf', weights2.is_leaf)
        
        # print('sensor_locations.requires_grad', sensor_locations.requires_grad)
        # print('sensor_locations.is_leaf', sensor_locations.is_leaf)
        
        params[0].data= mat2simplex(params[0]*gamma)
        params[1].data= mat2simplex(params[1]*gamma)
        
        # if tmp_loss - loss.item() > tols:
        #     tmp_loss = loss.item()
        # else:
        #     # print('Reached grad loss under tolerance ({})'.format(tols))
        #     break
        
        evolution['loss'].append(loss.item())
        evolution['sup_stability_const'].append(loss.item())
        
    plot_frames(measures, field_coordinates, it, frames_dir, prob_name, ker, cell_indices)
    
    it += 1
        
    return loss.cpu().detach().numpy()


def inf_stability_const_CBO(measures: torch.Tensor, 
                        field_coordinates: torch.Tensor,
                        prob_name,
                        params_opt = {'optimizer': 'Adam', 'lr': 0.001, 'nmax': 100,'type_loss':'W2', 'gamma':1}):        
    
    N = measures.shape[1]
    D = dimension(measures)
    
    field = field_coordinates.squeeze()
    
    no_of_sensors    = N+1
    sensor_locations = torch.linspace(field[0].item()*0.9, field[-1].item()*0.9, no_of_sensors).tolist()
    sensor_locations = torch.tensor([sensor_locations])
    # sensor_locations = torch.nn.Parameter(torch.tensor([sensor_locations], dtype=dtype, device=device))
    
    gamma = params_opt['gamma']
    niter = params_opt['nmax']
        
    S = torch.arange(N,device=device)
    inf_evolution = {'inf_sup_stability_const':[], 'sup_stability_const':[],'sensor_locations':[], 'iter':[]}
    
    tmp_loss = 1e20; tols = 1e-12
    
    frames_dir = check_create_dir(results_dir+'Frames_compute_mu_'+prob_name+'/')
    
    # Configure CBO parameters
    conf = {'alpha': 40.0,
        'dt': 0.01,
        'sigma': 0.7,#8,#5.1,#8.0,
        'lamda': 1.0,
        'batch_args':{
        'batch_size':1,
        'batch_partial': False},
        'd': no_of_sensors,
        'max_it': 1,
        'N': 5,
        'M': 1,
        'track_list': ['update_norm', 'energy','x', 'consensus', 'drift'],
        'resampling': False,
        'update_thresh': 0.002}

    # Define the initial positions of the particles
    x = cbx.utils.init_particles(shape=(conf['M'], conf['N'], conf['d']), x_min=field[0].item()*0.9, x_max = field[-1].item()*0.9)
    
    #DEBUG:
    # print('x', x)
    # print('x.shape', x.shape)
    # print('x.squeeze', np.squeeze(x))
    print('x.type', type(np.squeeze(x)))
    print('x to torch', torch.tensor(x.squeeze()[None, :]))

    # Define the CBO algorithm
    # f, sup_evolution, ker, cell_indices   = sup_stability_const_CBO(measures, field_coordinates, sensor_locations)
    
    # sup = f(measures, field_coordinates, x)
    it = 0
    sup = partial(f, measures, field_coordinates,
                  it, frames_dir, prob_name)
    
    dyn = CBO(sup, x=x, **conf)
    
    sched = scheduler(dyn, [multiply(name='alpha', factor=1.1, maximum=1e15)
                        #multiply(name='sigma', factor=1.005, maximum=6.)
                        ])
    # Run the CBO algorithm
    t = 0
    it = 0

    while not dyn.terminate():
        dyn.step()
        sched.update()
        
        # if it%10 == 0:
        print(dyn.f_min)
        print('Alpha: ' + str(dyn.alpha))
        print('Sigma: ' + str(dyn.sigma))
            
        it+=1
        # inf_evolution['sensor_locations'].append(sensor_locations.detach().cpu().clone())
        
        # print('sensor_locations.data', sensor_locations.data)
                
        ######## Plotting frames begins ########
        # fig, axs = plt.subplots(1, sharex=True, sharey=True)
        
        # sensor_placement = torch.zeros(field_coordinates.shape[0])
        # print('inf_evolution: ', inf_evolution['sensor_locations'])
        # indices = inf_evolution['sensor_locations']
        # indices = indices.tolist()
        
        # axs.plot(field.cpu().numpy(), target_field.cpu().squeeze().numpy(), 'black')
        
        # axs.plot(field_coordinates.squeeze().cpu().numpy(),bary.cpu().detach().squeeze().numpy())
        
        # for i in range(measures.shape[1]): 
        #     axs.plot(field.cpu().numpy(), torch.squeeze(measures.cpu())[i].numpy())
        
        # axs.plot(field.cpu().numpy(), sensor_placement.cpu().numpy(), markevery = cell_indices, ls = "", 
        #           marker="o", markerfacecolor='blue', markersize=10)
        
        # axs.grid(False)
        
        # axs.set_xlabel('x')
        # axs.set_ylabel('Observables')
        # axs.set_ylabel(r'$\mu_{normalized}$', rotation=90)
        # axs.legend([r'Target', 'Bary approx'],loc='best')
        
        # axs2 = axs.twinx()
        
        # for k,l in enumerate(cell_indices): 
        #     tmp = ker[l][:].cpu().detach().numpy()
        #     axs2.plot(field_coordinates.squeeze().cpu().numpy(), tmp, 'blue', alpha = 0.1, 
        #               linestyle='dotted')
        #     axs2.fill_between(field_coordinates.squeeze().cpu().numpy(), tmp, 0, color='blue', alpha=.03)
                
        # axs2.grid(False)
                
        # fig.savefig(frames_dir+'frame_{}.png'.format(iter))
        
        # plt.close()        
                    
        #DEBUG
        # print('sensor_locations.requires_grad', sensor_locations.requires_grad)
        # print('sensor_locations.is_leaf', sensor_locations.is_leaf)
        
        # sensor_locations.data.clamp_(field[0].item(), field[-1].item())
        
        # inf_evolution['iter'].append(iter)
        # inf_evolution['inf_sup_stability_const'].append(inf_loss.item())
        # inf_evolution['sup_stability_const'].append(sup_evolution['sup_stability_const'][-1])
        
    # plot_gif(iter, frames_dir, prob_name)
        
    # return inf_loss, inf_evolution, sup_evolution
    # return inf_evolution, sup_evolution
    
def plot_frames(measures,field_coordinates,iter,
                frames_dir,prob_name,ker,cell_indices):
    print('Here plots')
    
    fig, axs = plt.subplots(1, sharex=True, sharey=True)
    
    sensor_placement = np.zeros(field_coordinates.shape[0])

    for i in range(measures.shape[1]): 
        axs.plot(field_coordinates.cpu().numpy(), torch.squeeze(measures.cpu())[i].numpy())
    
    axs.plot(field_coordinates.cpu().numpy(), sensor_placement, markevery = cell_indices, ls = "", 
              marker="o", markerfacecolor='blue', markersize=10)
    
    axs.grid(False)
    
    axs.set_xlabel('x')
    axs.set_ylabel(r'$\mu_{normalized}$', rotation=90)
    axs.legend([r'Target', 'Bary approx'],loc='best')
    
    axs2 = axs.twinx()
    
    for k,l in enumerate(cell_indices): 
        tmp = ker[l][:].cpu().detach().numpy()
        axs2.plot(field_coordinates.squeeze().cpu().numpy(), tmp, 'blue', alpha = 0.1, 
                  linestyle='dotted')
        axs2.fill_between(field_coordinates.squeeze().cpu().numpy(), tmp, 0, color='blue', alpha=.03)
            
    axs2.grid(False)
            
    fig.savefig(frames_dir+'frame_{}.png'.format(iter))
    
    plt.close()
    
    plot_gif(iter, frames_dir, prob_name)
    
    
class consensus_opt:
    def __init__(self, target: torch.Tensor, measures: torch.Tensor,
                 field_coordinates: torch.Tensor,
                 prob_name,
                 params_opt = {'optimizer': 'Adam', 'lr': 0.01, 'nmax': 30,'type_loss':'W2', 'gamma':1}):
        
        self.target   = target
        self.measures = measures
        self.field_coordinates = field_coordinates
        self.prob_name = prob_name
        self.params_opt = params_opt
        self.N = measures.shape[1]
        self.D = dimension(measures)
        self.no_of_sensors    = self.N
        
        self.gamma = self.params_opt['gamma']
        self.niter = self.params_opt['nmax']
        
        # self.variance = 0.05
        field         = self.field_coordinates.squeeze()
        # self.variance = (field[-1]-field[0])*0.015
        self.variance = (field[-1]-field[0])*0.03
        self.k        = 1
        self.wts      = []
        # (field[-1]-field[0])*0.03125
        
        conf = {'alpha': 50.0,
            'dt': 0.1,
            # 'max_time': 5.0,
            'sigma': 1.0, #self.k*(3.0+self.variance),
            'lamda': 1.0,
            # 'batch_args':{
            # 'batch_size':200,
            # 'batch_partial': False},
            'd': self.no_of_sensors,
            'max_it': 100,
            'N': 100,
            'M': 2,
            'track_list': ['update_norm', 'energy','x', 'consensus', 'drift'],
            'resampling': False,
            'update_thresh': 0.002}
        
        self.conf = conf
                
        # self.frames_dir = check_create_dir(results_dir+'Compute_C_s_CBO_'+prob_name+'/')
        self.frames_dir = frames_dir = check_create_dir(results_dir+'Compute_C_s_'+prob_name+'/'+'five_measures/CBO/')
        # self.frames_dir = frames_dir = check_create_dir(results_dir+'Compute_C_s_two_measures/CBO/')
        # self.frames_dir = check_create_dir(results_dir+'Compute_C_s_CBO_translated_meas_k'+str(self.k)+'_'+prob_name+'/')
        # self.frames_dir = check_create_dir(results_dir+'Reconstruct_'+prob_name+'/'+'Trial/')

        self.iter = 0
        
    def convolution_kernel_v2(self, field_coordinates, sensor_locations):
        
        field = field_coordinates.squeeze()
        
        tmp_field = field
        tmp_sensor_locations = sensor_locations
        
        if sensor_locations.dim() == 3:          
        
            M = sensor_locations.shape[0]
            N = sensor_locations.shape[1]
            d = sensor_locations.shape[-1]
    
            tmp_field = tmp_field.repeat(M,N,d,1)
                        
            tmp_sensor_locations = tmp_sensor_locations.reshape(M,N,d,1)
        
        else:
            
            print('sensor_locations.dim()',sensor_locations.dim())
            
            d = sensor_locations.shape[-1]
                        
            tmp_field = tmp_field.repeat(d,1)
            
            tmp_sensor_locations = tmp_sensor_locations.reshape(d,1)
        
        
        npoints = field.shape[-1]
        
        # h   = abs(field[0].item()-field[1].item())
        
        cell_indices = []
        # sigma = 0.25  #increase for higher influence from the neighbours
        # sigma = (field[-1]-field[0])*0.03125
        variance = self.variance
        
        # print(field.get_device(), sensor_locations.get_device())
        
        gaussian = lambda x : (torch.exp(-0.5*((tmp_field-x)/variance)**2)/((variance*(2*math.pi)**0.5)))    
        
        ker = gaussian(tmp_sensor_locations) #; print('ker.shape',ker.shape)
        
        return ker, cell_indices
    
    def optimize_over_wts(self,x):

        measures = self.measures
        params_opt = self.params_opt
        N = self.N
        
        sensor_locations = torch.tensor(x, device=device)
        
        no_of_sensors = sensor_locations.shape[-1]
        # no_of_sensors    = 6
        # sensor_locations = [[-1.5, 0.0, 0.5, 1.1, 1.6, 1.9]]
        # sensor_locations = torch.tensor(sensor_locations)

        x        = torch.linspace(-0.5, 0.5, N)
        gaussian = lambda x : torch.exp(-0.5*((x)/2)**2)/((2*(2*math.pi)**0.5))
        g        = gaussian(x)/gaussian(x).sum()
        
        weights  = torch.nn.Parameter(torch.ones((1,N), dtype=dtype, device=device)/N)
        # weights = torch.nn.Parameter(torch.tensor([g.tolist()], dtype=dtype, device=device))
        
        gamma = params_opt['gamma']
        niter = params_opt['nmax']
        
        def get_optimizer(weights):
            if params_opt['optimizer'] == 'Adam':
                return torch.optim.Adam([weights], lr=params_opt['lr'], maximize=False)
            else:
                return torch.optim.SGD([weights], lr=params_opt['lr'], momentum=0.9, maximize=False)
            
        optimizer = get_optimizer(weights)
        scheduler = ExponentialLR(optimizer, gamma=0.9)
        
        self.ker, cell_indices = self.convolution_kernel_v2(self.field_coordinates, sensor_locations)
        
        S = torch.arange(N,device=device)
        evolution = {'loss':[],'weights':[],'sup_stability_const':[]}
        
        tmp_loss = 1e20; tols = 1e-18

        for iter in tqdm(range(niter)):
            evolution['weights'].append(weights.detach().clone())
            
            optimizer.zero_grad()
            
            bary = ImagesBarycenter_1d(measures=measures, weights=weights)
            
            observation_target = torch.matmul(self.ker,self.target.squeeze())
            observation_bary   = torch.matmul(self.ker,bary.squeeze())
            
            W2 = ImagesLoss(self.target, bary) #Squared W_2
            
            # W2_Lp = tau*ImagesLoss(bary1, bary2, scaling=0.8) 
            # + (1.-tau)*torch.norm((bary1-bary2), p = 2, dim = -1)
            
            L2 = torch.nn.MSELoss()(observation_target[None, None, :],
                                      observation_bary[None, None, :])
            
            # L2 = torch.norm((observation_target-observation_bary), p=2, dim = -1)
            
            loss = W2/L2
            
            loss.backward(retain_graph=True)
            optimizer.step()
            # scheduler.step()
            
            #DEBUG
            # print('weights1.requires_grad', weights1.requires_grad)
            # print('weights2.requires_grad', weights2.requires_grad)
            
            # print('weights1.is_leaf', weights1.is_leaf)
            # print('weights2.is_leaf', weights2.is_leaf)
            
            # print('sensor_locations.requires_grad', sensor_locations.requires_grad)
            # print('sensor_locations.is_leaf', sensor_locations.is_leaf)
            
            weights.data= mat2simplex(weights*gamma)
            
            # if tmp_loss - loss.item() >= tols:
            #     tmp_loss = loss.item()
            # else:
            #     # print('Reached grad loss under tolerance ({})'.format(tols))
            #     break
            
            evolution['loss'].append(loss.item())
            # evolution['sup_stability_const'].append(1./(loss.item()**0.5))
            evolution['sup_stability_const'].append(loss.item()**0.5)
            
        self.bary1 = self.target.detach().clone()
        self.bary2 = bary.detach().clone()
        
        tmp = torch.nn.MSELoss(reduction='none')
        L2  = torch.sum(tmp(observation_target,observation_bary),dim=-1)
        inf_loss = W2/L2
            
        return inf_loss.cpu().detach().numpy()
        
    def sup(self,x):
        measures = self.measures
        params_opt = self.params_opt
        N = self.N
        
        sensor_locations = torch.tensor(x, device=device)
        
        ###### Recomputing Kernel for best particle in order to plot #######
        
        no_of_sensors = sensor_locations.shape[-1] #; print('No. of sensors', no_of_sensors)

        tmp        = torch.linspace(-0.5, 0.5, N)
        gaussian = lambda tmp : torch.exp(-0.5*((tmp)/2)**2)/((2*(2*math.pi)**0.5))
        g        = gaussian(tmp)/gaussian(tmp).sum()
        
        weights1 = torch.nn.Parameter(torch.ones((1,N), dtype=dtype, device=device)/N)
        # weights2 = torch.nn.Parameter(torch.tensor([[0.8,0.2]], dtype=dtype, device=device))
        weights2 = torch.nn.Parameter(torch.tensor([g.tolist()], dtype=dtype, device=device))
        
        gamma = params_opt['gamma']
        niter = params_opt['nmax']
        
        def get_optimizer(params):
            if params_opt['optimizer'] == 'Adam':
                return torch.optim.Adam(params, lr=params_opt['lr'], maximize=True)
            else:
                return torch.optim.SGD(params, lr=params_opt['lr'], momentum=0.9, maximize=True)
        
        params = [weights1, weights2]
        # params = self.wts
        optimizer = get_optimizer(params)
        scheduler = ExponentialLR(optimizer, gamma=0.99)
        
        self.ker, cell_indices = self.convolution_kernel_v2(self.field_coordinates, sensor_locations)
        
        S = torch.arange(N,device=device)
        evolution = {'loss':[],'weights1':[],'weights2':[], 'sup_stability_const':[]}
        
        tmp_loss = 1e20; tols = 1e-18

        # for iter in tqdm(range(niter)):
        for iter in range(niter):
            evolution['weights1'].append(params[0].detach().clone())
            evolution['weights2'].append(params[1].detach().clone())
            
            optimizer.zero_grad()
            
            bary1 = ImagesBarycenter_1d(measures=measures, weights=params[0])
            bary2 = ImagesBarycenter_1d(measures=measures, weights=params[1])
            
            # bary1 = bary1.repeat()
            
            # print(bary1.shape, bary2.shape)
            
            observation_bary1 = torch.matmul(self.ker,bary1.squeeze())
            observation_bary2 = torch.matmul(self.ker,bary2.squeeze())
            
            #DEBUG:
            # print('Obs shapes', observation_bary1.shape, observation_bary2.shape)
            
            W2   = ImagesLoss(bary1, bary2) #Squared W_2
            
            # L2 = torch.nn.MSELoss()(observation_bary1[None, None, :],observation_bary2[None, None, :])
            L2 = torch.nn.MSELoss()(observation_bary1,observation_bary2)
            # tmp = torch.nn.MSELoss(reduction='none')
            # L2  = torch.sum(tmp(observation_bary1,observation_bary2),dim=-1)            
            
            # loss = torch.pow((Lp/W2), 0.5)
            # loss = torch.pow((W2/Lp), 0.5)
            # loss = Lp/W2
            # loss = (W2**0.5)/Lp
            loss = W2
            
            # print(loss.shape)
            
            loss.backward()
            optimizer.step()
            scheduler.step()
                        
            params[0].data= mat2simplex(params[0]*gamma)
            params[1].data= mat2simplex(params[1]*gamma)
            
            # if tmp_loss - loss.item() > tols:
            #     tmp_loss = loss.item()
            # else:
            #     # print('Reached grad loss under tolerance ({})'.format(tols))
            #     break
            
            evolution['loss'].append(loss.item())
            evolution['sup_stability_const'].append(loss.item())
            
        # self.plot_frames()
        
        # self.iter += 1
        
        self.wts = params
        
        self.bary1 = bary1.detach().clone()
        self.bary2 = bary2.detach().clone()
        
        ######## Plotting barycenters DEBUG ########
        fig, axs = plt.subplots(1, sharex=True, sharey=True)
        
        for i in range(self.measures.shape[1]): 
            axs.plot(self.field_coordinates.cpu().numpy(), torch.squeeze(self.measures.cpu())[i].numpy())                
        
        axs.plot(self.field_coordinates.cpu().numpy(), bary1.cpu().detach().squeeze().numpy())
        axs.plot(self.field_coordinates.cpu().numpy(), bary2.cpu().detach().squeeze().numpy())
        
        axs.set_xlabel('x')
        axs.set_title(r'Barycenters')
        
        # axs[1].plot(self.field_coordinates.cpu().numpy(), observation_bary1.cpu().detach().squeeze().numpy())
        # axs[1].plot(self.field_coordinates.cpu().numpy(), observation_bary2.cpu().detach().squeeze().numpy())
        # axs[1].set_title(r'Observations')
        
        fig.savefig(self.frames_dir+'barycenter_plot_{}.png'.format(self.iter))
        
        plt.close()
        
        ############################################
        
        # MSE = torch.nn.MSELoss(reduction='none')
        # L2  = torch.sum(MSE(observation_bary1,observation_bary2),dim=-1)
        # inf_loss = W2/L2
            
        # return L2.cpu().detach().numpy()
        
        #DEBUG:
        L2 = torch.linalg.norm((observation_bary1-observation_bary2), axis=-1)
        print('L2.shape, loss.shape', L2.shape, loss.shape)
        return L2.cpu().detach().numpy()
    
    def plot_frames(self, x, niter, it, nfac):
        # print('Here plots')
        
        m = x.shape[-1] #; print('m', m)
        
        sensor_locations_as_list = x.tolist()[-1]
        field = self.field_coordinates.squeeze()
        h     = abs(field[0].item()-field[1].item())
                
        cell_indices = []
        print('sensor_locations_as_list',sensor_locations_as_list)
        for i,j in enumerate(sensor_locations_as_list):
            
            if j < field[0].item():
                j = field[-1].item()
            elif j > field[-1].item():
                j = field[0].item()
            else:
                pass        
        
            # index = int(abs(field_as_list[0]-j)//h)
            index = int((j-field[0].item())//h)
            
            # print('index',index)
            
            # if index >= field.shape[-1]:
            #     index = 0
            # elif index < 0:
            #     index = field.shape[-1]
            # else:
            #     pass
            
            cell_indices.append(index)
            
        best_particle_ker = torch.zeros(m, field.shape[-1], dtype = dtype, device = device)
        variance = self.variance
        
        field_repeat = field.repeat(m,1)
        gaussian = lambda x : (torch.exp(-0.5*((field_repeat-x)/variance)**2)/((variance*(2*math.pi)**0.5)))
        x_torch   = torch.tensor(x[-1], device=device); print(x_torch)
        x_reshape = x_torch[None, :].reshape(m,1); print('x_reshape.shape', x_reshape.shape) #DEBUG
        best_particle_ker = gaussian(x_reshape)
        
        fig, axs = plt.subplots(2, sharex=True, sharey=True)
        
        sensor_placement = np.zeros(self.field_coordinates.shape[0])

        for i in range(self.measures.shape[1]): 
            axs[0].plot(self.field_coordinates.cpu().numpy(), torch.squeeze(self.measures.cpu())[i].numpy())
        
        axs[0].plot(self.field_coordinates.cpu().numpy(), sensor_placement, markevery = cell_indices, ls = "", 
                  marker="o", markerfacecolor='blue', markersize=10)
        
        axs[0].grid(False)
        
        axs[0].set_xlabel('x')
        
        axs[1].plot(field.cpu().numpy(), self.bary1.cpu().squeeze().numpy())
        axs[1].plot(field.cpu().numpy(), self.bary2.cpu().squeeze().numpy())
        axs[1].plot(field.cpu().numpy(), sensor_placement, markevery = cell_indices, ls = "", 
                  marker="o", markerfacecolor='blue', markersize=10)
        # axs[1].grid(False)
        axs[1].set_xlabel('x')
        axs[1].legend([r'Barycenter 1', 'Barycenter 2'],loc='best')
        
        axs2 = axs[1].twinx()   
        
        # for k,l in enumerate(self.cell_indices):
        for l in range(self.no_of_sensors*(nfac+1)):
            # tmp = self.ker[l][:].cpu().detach().numpy()
            tmp = best_particle_ker[l][:].cpu().detach().numpy()
            axs2.plot(self.field_coordinates.squeeze().cpu().numpy(), tmp, 'blue', alpha = 0.1, 
                      linestyle='dotted')
            axs2.fill_between(self.field_coordinates.squeeze().cpu().numpy(), tmp, 0, color='blue', alpha=.03)
                
        axs2.grid(False)
                
        fig.savefig(self.frames_dir+'frame_{}.png'.format(it))
        
        if it == 0 or it == int(self.conf['max_it']*0.25) or iter == int(self.conf['max_it']*0.5) or it == self.conf['max_it']-1:
            fig.savefig(self.frames_dir+'Nsensors_{}n_frame_{}.png'.format(nfac+1,it))
        
        plt.close()
        
        # plot_gif(self.iter, self.frames_dir, self.prob_name)
        
    def plot_gif(self, iterations, frames_dir, prob_name, nfac):
        frames = []
        
        for i in range(iterations):
            image = imageio.v2.imread(frames_dir+'frame_{}.png'.format(i))
            frames.append(image)
            
        imageio.mimsave(frames_dir+prob_name+str(nfac)+'.gif', frames, duration=5, loop = 1)
        
    def plot_graphs(self, niter, nfac, C_s, frames_dir, prob_name):
        
        fig, axs = plt.subplots(1, sharex=True, sharey=True)

        x = np.arange(0, niter)
        
        # for i in range(nfac): axs.plot(x, C_s[i])
        axs.plot(x, C_s[0])
        axs.plot(x, C_s[1])
        axs.plot(x, C_s[2])
        axs.set_xlabel('niter')
        # axs.set_ylabel(r'$C_L$', rotation=90)
        # axs.set_ylabel(r'$C_S$', rotation=90)
        plt.yscale("log")
        axs.legend([r'm=n', 'm=2n', 'm=3n'],loc='best')

        fig.savefig(frames_dir+prob_name)
        
    def plot_sensor_trajectory(self, no_of_sensors,
                               sensors_history, prob_name):
    
        fig, axs = plt.subplots(1, sharex=True, sharey=True)
    
        x = np.arange(0, self.conf['max_it'])
        
        for i in range(no_of_sensors): axs.plot(x, sensors_history[i])
    
        # axs.legend([r'sensor {}'.format(i+1) for i in range(no_of_sensors)],loc='best')
        axs.set_xlabel('niter')
        axs.set_ylabel(r'$x$', rotation=90)
    
        fig.savefig(self.frames_dir+'sensor_history'+'_'+str(no_of_sensors)+'.eps', format='eps')
    
    def inf(self):
        
        # C_s = np.zeros(3*self.conf['max_it'])
        # C_s = C_s.reshape(3,self.conf['max_it'])
        
        config = self.conf
        
        Filename = self.frames_dir+'Compute_C_s_CBO_'+str(config['M'])+'_'+str(config['N'])+'_'+str(config['d'])
        
        for nfac in range(3):
            
            x        = torch.linspace(-0.5, 0.5, self.N)
            gaussian = lambda x : torch.exp(-0.5*((x)/2)**2)/((2*(2*math.pi)**0.5))
            g        = gaussian(x)/gaussian(x).sum()
        
            weights1 = torch.nn.Parameter(torch.ones((1,self.N), dtype=dtype, device=device)/self.N)
            # weights2 = torch.nn.Parameter(torch.tensor([[0.5,0,0.5]], dtype=dtype, device=device))
            weights2 = torch.nn.Parameter(torch.tensor([g.tolist()], dtype=dtype, device=device))
            
            self.wts = [weights1, weights2]
            
            print('******** m = {}n ********'.format(nfac+1))
        
            # S = torch.arange(self.N,device=device)
            # inf_evolution = {'inf_sup_stability_const':[], 'sup_stability_const':[],'sensor_locations':[], 'iter':[]}
            
            inf_loss, tmp, sensors_history = self.compute_CBO(config, nfac)
            
            df = pd.DataFrame(sensors_history)
            df.to_csv(self.frames_dir+'sensor_history_'+str((nfac+1)*config['d']), index=False)
            print('Sensor locations written to file.')
            self.plot_sensor_trajectory((nfac+1)*config['d'], sensors_history, self.prob_name)
            
            if nfac == 0:
                C_s = tmp
            else:
                C_s = np.concatenate((C_s, tmp), axis=0)
                        
            self.plot_gif(config['max_it'], self.frames_dir, self.prob_name, nfac)
            print('GIF created')
            
            file = open(Filename+'_'+str(nfac)+'.txt', "w")
            for i in inf_loss: file.write("%s\n" % i)
            file.close()
            print('Writing inf_loss to file complete')
            
        self.plot_graphs(self.conf['max_it'], nfac, C_s, self.frames_dir, self.prob_name)
        print('C_s graph plotted. Exiting')
        
        self.reconstruct_target_w_static_sensors()
        
    def reconstruct_target_w_static_sensors(self):
                           
        params_opt = self.params_opt
        frames_dir = check_create_dir(self.frames_dir+'/'+'reconstruct_target/')
        
        field = self.field_coordinates.squeeze()
        
        for nfac in range(3):
            
            print('***** m = {}n'.format(nfac+1))
            
            no_of_sensors = (nfac+1)*self.conf['d']
            
            # sensor_locations = torch.linspace(field[0].item()*0.9, field[-1].item()*0.9, no_of_sensors).tolist(); arange = 'even'
            sensor_locations = torch.tensor(torch.load(self.frames_dir+'{}_sensor-coordinates'.format(no_of_sensors)), device=device)
            print('x',sensor_locations)
            
            x        = torch.linspace(-0.5, 0.5, self.N)
            gaussian = lambda x : torch.exp(-0.5*((x)/2)**2)/((2*(2*math.pi)**0.5))
            wts      = gaussian(x)/gaussian(x).sum()
            weights  = torch.nn.Parameter(torch.tensor([wts.tolist()], dtype=dtype, device=device))
            
            gamma = params_opt['gamma']
            niter = params_opt['nmax']
            
            def get_optimizer(weights):
                if params_opt['optimizer'] == 'Adam':
                    return torch.optim.Adam([weights], lr=params_opt['lr'], maximize=False)
                else:
                    return torch.optim.SGD([weights], lr=params_opt['lr'], momentum=0.9, maximize=False)
                
            optimizer = get_optimizer(weights)
            scheduler = ExponentialLR(optimizer, gamma=0.99)
            
            ker, cell_indices = self.convolution_kernel_v2(self.field_coordinates, sensor_locations[None, :])
            
            S = torch.arange(self.N,device=device)
            evolution = {'loss':[], 'W2_loss':[],'weights':[],'C_s':[]}
            
            Filename = frames_dir+'batch_wts_'+str(no_of_sensors)

            for iter in tqdm(range(niter)):
                evolution['weights'].append(weights.detach().clone())
                
                optimizer.zero_grad()
                
                bary = ImagesBarycenter_1d(measures=self.measures, weights=weights)
                
                observation_target = torch.matmul(ker,self.target.squeeze())
                observation_bary   = torch.matmul(ker,bary.squeeze())
                
                W2 = ImagesLoss(self.target, bary) #Squared W_2
                
                L2 = torch.nn.MSELoss()(observation_target[None, None, :],
                                          observation_bary[None, None, :])
                
                # L2 = torch.norm((observation_target-observation_bary), p=2, dim = -1)
                
                loss = L2                                
                
                weights.data= mat2simplex(weights*gamma)
                
                # if tmp_loss - loss.item() >= tols:
                #     tmp_loss = loss.item()
                # else:
                #     # print('Reached grad loss under tolerance ({})'.format(tols))
                #     break
                            
                ######## Plotting frames begins ########
                fig, axs = plt.subplots(2, sharex=True, sharey=True)
                
                sensor_placement = torch.zeros(self.field_coordinates.shape[0])
                
                for i in range(self.measures.shape[1]): 
                    axs[0].plot(field.cpu().numpy(), torch.squeeze(self.measures.cpu())[i].numpy())
                
                axs[0].plot(field.cpu().numpy(), sensor_placement.cpu().numpy(), markevery = cell_indices, ls = "", 
                          marker="o", markerfacecolor='blue', markersize=10)
                
                axs[0].grid(False)
                
                axs[0].set_xlabel('x')
                axs[0].set_title(r'Reconstructing target using observables')
                
                axs[1].plot(field.cpu().numpy(), self.target.cpu().detach().squeeze().numpy())
                axs[1].plot(field.cpu().numpy(), bary.cpu().detach().squeeze().numpy())
                axs[1].plot(field.cpu().numpy(), sensor_placement.cpu().numpy(), markevery = cell_indices, ls = "", 
                          marker="o", markerfacecolor='blue', markersize=10)
                # axs[1].grid(False)
                axs[1].set_xlabel('x')
                axs[1].legend([r'Target', 'Barycenter'],loc='best')
                # axs[1].title(r'$\inf_{x\in\Omega}\, \sup_{\Lambda_1,\Lambda_2} \frac{W_2(\mu,\nu)}{MSE(\mu,\nu)}$')
                
                axs2 = axs[1].twinx()
                
                for l in range(no_of_sensors): 
                    tmp = ker[l][:].cpu().detach().numpy()
                    axs2.plot(self.field_coordinates.squeeze().cpu().numpy(), tmp, 'blue', alpha = 0.1, 
                              linestyle='dotted')
                    axs2.fill_between(self.field_coordinates.squeeze().cpu().numpy(), tmp, 0, color='blue', alpha=.03)
                        
                axs2.grid(False)
                
                fig.savefig(frames_dir+'frame_{}.png'.format(iter))
                
                if iter == 0 or iter == int(niter*0.25) or iter == int(niter*0.5) or iter == niter-1:
                    fig.savefig(frames_dir+'Nsensors_{}n_frame_{}.png'.format(nfac+1,iter))
                
                # plt.title(r"Hello \Tex", fontsize=6)
                plt.close()        
                
                evolution['loss'].append(loss.item())
                evolution['W2_loss'].append(W2.item())
                evolution['weights'].append(weights.detach().squeeze().clone())
                evolution['C_s'].append(W2/L2)
                
                loss.backward(retain_graph=True)
                optimizer.step()
                scheduler.step()

            
            prob_name_gif = str('Reconstruct_')+self.prob_name+str('_{}_sensors'.format(no_of_sensors))
            
            self.plot_gif(iter, frames_dir, prob_name_gif, nfac)
                
            Filename = frames_dir+'C_s_reconstruct_'+self.prob_name+str('_{}_sensors'.format(no_of_sensors))
            file = open(Filename+'.txt', "w")
            # for i in inf_evolution['inf_sup_stability_const']: file.write("%s\n" % i)
            for i in evolution['loss']: file.write("%s\n" % i)
            file.close()
            print('Writing C_s to file complete')
            
            Filename = frames_dir+'W2_reconstruct_'+self.prob_name+str('_{}_sensors'.format(no_of_sensors))
            file = open(Filename+'.txt', "w")
            # for i in inf_evolution['inf_sup_stability_const']: file.write("%s\n" % i)
            for i in evolution['W2_loss']: file.write("%s\n" % i)
            file.close()
            print('Writing W2 to file complete')
            
            if nfac == 0:
                C_s     = np.array([ evolution['loss']])
                W2_loss = np.array([ evolution['W2_loss']])
            else:
                tmp  = np.array([ evolution['loss']])
                tmp2 = np.array([ evolution['W2_loss']])
                C_s     = np.concatenate((C_s, tmp), axis=0)
                W2_loss = np.concatenate((W2_loss, tmp2), axis=0)
                print('shape C_s', C_s.shape)
                
        self.plot_graphs(niter, nfac, C_s, frames_dir, self.prob_name+'_C_s')
        self.plot_graphs(niter, nfac, W2_loss, frames_dir, self.prob_name+'_W2_loss')
    
    def compute_CBO(self, config, nfac):            
        
        # Define the initial positions of the particles
        x = cbx.utils.init_particles(shape=(config['M'], config['N'], config['d']*(nfac+1)),
                                      x_min=self.field_coordinates[0].item()*0.9, x_max = self.field_coordinates[-1].item()*0.9)
        
        inf_loss = []
        
        # Run the CBO algorithm
        it = 0
        t  = 0
        C_s = np.zeros(config['max_it'])
        
        dyn = CBO(self.sup, x=x, f_dim='3D', noise='anisotropic', **self.conf)
                    
        sched = scheduler(dyn, [multiply(name='alpha', factor=1.1, maximum=1e15),
                            #multiply(name='sigma', factor=1.005, maximum=6.)
                            ])
        
        while not dyn.terminate():                                    
            
            print('Inf Loss', dyn.f_min)
            print('Infimum iteration #',it)
            print('x',dyn.x)
            print('best_particles', dyn.best_particle[0], dyn.best_particle[1])
            inf_loss.append(dyn.f_min)
            # print('*************************')            
            
            # if it%5 == 0:                
            self.plot_frames(dyn.best_particle, config['max_it'], it, nfac)
            print('Frames plotted')
            
            tmp = dyn.best_particle[0]
            
            if it == 0:
                # C_s = np.array([dyn.f_min[0]])
                sensors_history = tmp.reshape((nfac+1)*config['d'],1)
            else:
                # C_s = np.concatenate((C_s, np.array([dyn.f_min[0]])), axis=0)
                sensors_history = np.concatenate((sensors_history,
                                                     tmp.reshape((nfac+1)*config['d'],1)), axis=1)
            
            C_s[it] = dyn.f_min[0]
            
            self.iter += 1
            it+=1
            
            dyn.step()
            sched.update()
            
        torch.save(dyn.best_particle[1], self.frames_dir+'{}_sensor-coordinates'.format(config['d']*(nfac+1)))    

        return inf_loss, C_s[None, :], sensors_history