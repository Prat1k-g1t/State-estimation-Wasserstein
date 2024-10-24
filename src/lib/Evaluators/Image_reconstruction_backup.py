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
                       params_opt = {'optimizer': 'Adam', 'lr': 0.01, 'nmax': 100,'type_loss':'W2', 'gamma':1}):
    
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
                       params_opt = {'optimizer': 'AdamW', 'lr': 0.01, 'nmax': 100,'type_loss':'W2', 'gamma':1, 'k_sparse':5}):
    
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
        
        if nfac == 0:
            no_of_sensors = (nfac+2)*N
        else:
            no_of_sensors = (nfac+1)*N
        
        # sensor_locations = torch.linspace(field[0].item()*0.9, field[-1].item()*0.9, no_of_sensors).tolist(); arange = 'even'
        sensor_locations = torch.load(saved_dir+'{}_{}_sensor-coordinates'.format(no_of_sensors,arange))
        sensor_locations = sensor_locations.to(device)
        
        x        = torch.linspace(-0.5, 0.5, N)
        gaussian = lambda x : torch.exp(-0.5*((x)/2)**2)/((2*(2*math.pi)**0.5))
        wts      = gaussian(x)/gaussian(x).sum()
        weights  = torch.nn.Parameter(torch.tensor([wts.tolist()], dtype=dtype, device=device))
        
        gamma  = params_opt['gamma']
        niter  = params_opt['nmax']
        n_keep = params_opt['k_sparse']
        
        def get_optimizer(weights):
            if params_opt['optimizer'] == 'Adam':
                return torch.optim.Adam([weights], lr=params_opt['lr'], maximize=False)
            elif params_opt['optimizer'] == 'AdamW':
                return torch.optim.Adam([weights], lr=params_opt['lr'], maximize=False)
            else:
                return torch.optim.SGD([weights], lr=params_opt['lr'], nesterov=False,
                                       momentum=0., maximize=False)
            
        optimizer = get_optimizer(weights)
        scheduler = ExponentialLR(optimizer, gamma=0.9)
        
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
            
            # L2 = torch.nn.MSELoss()(observation_target[None, None, :],
            #                           observation_bary[None, None, :])
            
            L2 = no_of_sensors*torch.nn.MSELoss()(observation_target[None, None, :],
                                      observation_bary[None, None, :])
            
            loss = L2
            
            loss.backward()
            optimizer.step()
            scheduler.step()
            
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

def compute_barycenters(measures: torch.Tensor,
                        params_opt = {'optimizer': 'Adam', 'lr': 0.01, 'nmax': 50,'type_loss':'W2', 'gamma':1}):

    # eps = 1e-3; tau = 1.
    # Loss = SamplesLoss("sinkhorn", blur = eps, scaling=.9, debias=True)
    
    N = measures.shape[1]
    D = dimension(measures)

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
    
    params = [weights1, weights2]
    # params = wts
    optimizer = get_optimizer(params)
    scheduler = ExponentialLR(optimizer, gamma=0.99)
    
    # ker, cell_indices = convolution_kernel_v2(field_coordinates, sensor_locations)
       
    S = torch.arange(N,device=device)
    evolution = {'loss':[],'weights1':[],'weights2':[], 'sup_stability_const':[]}
    
    tmp_loss = 1e20; tols = 1e-18

    for iter in tqdm(range(niter)):
        evolution['weights1'].append(params[0].detach().clone())
        evolution['weights2'].append(params[1].detach().clone())
        
        optimizer.zero_grad()
        
        bary1 = ImagesBarycenter_1d(measures=measures, weights=params[0])
        bary2 = ImagesBarycenter_1d(measures=measures, weights=params[1])
        
        # observation_bary1 = torch.matmul(ker,bary1.squeeze())
        # observation_bary2 = torch.matmul(ker,bary2.squeeze())
        
        # observation_bary1 = torch.matmul(bary1.squeeze(),ker)
        # observation_bary2 = torch.matmul(bary2.squeeze(),ker)
        
        loss   = ImagesLoss(bary1, bary2) #Squared W_2
        
        # W2_Lp = tau*ImagesLoss(bary1, bary2, scaling=0.8) 
        # + (1.-tau)*torch.norm((bary1-bary2), p = 2, dim = -1)
        
        # L2 = (torch.nn.MSELoss()(observation_bary1[None, None, :],
        #                          observation_bary2[None, None, :]))
        
        # Lp = torch.norm((observation_bary1-observation_bary2), p=2, dim = -1)
        
        # loss = torch.pow((Lp/W2), 0.5)
        # loss = torch.pow((W2/Lp), 0.5)
        # loss = W2/(L2+eps)
        # loss = L2/W2
        
        loss.backward()
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
        
    return loss.detach(), evolution, bary1.detach(), bary2.detach(), params

def compute_Cs(measures, field_coordinates, wts, sensor_locations):
    
    # N = self.N
    
    ker, cell_indices = convolution_kernel_v2(field_coordinates, sensor_locations)
    
    bary1 = ImagesBarycenter_1d(measures=measures, weights=wts[0])
    bary2 = ImagesBarycenter_1d(measures=measures, weights=wts[1])
    
    observation_bary1 = torch.matmul(ker,bary1.squeeze())
    observation_bary2 = torch.matmul(ker,bary2.squeeze())
            
    W2 = ImagesLoss(bary1, bary2) #Squared W_2
            
    L2 = torch.nn.MSELoss()(observation_bary1,observation_bary2)
    
    return W2/L2, bary1, bary2, W2.item(), L2, ker, cell_indices
    
def compute_Cs_4_CBO(measures, field_coordinates, wts, sensor_locations):
    #Dedicated routine for CBO in order to receive the correct return shape of loss func.
    
    # N = self.N
    
    ker, cell_indices = convolution_kernel_v2(field_coordinates, sensor_locations)
    
    bary1 = ImagesBarycenter_1d(measures=measures, weights=wts[0])
    bary2 = ImagesBarycenter_1d(measures=measures, weights=wts[1])
    
    # print('Shape of bary1', bary1.shape)
    # print('Shape of bary2', bary2.shape)
    
    observation_bary1 = torch.matmul(ker,bary1.squeeze())
    observation_bary2 = torch.matmul(ker,bary2.squeeze())
            
    W2 = ImagesLoss(bary1, bary2) #Squared W_2
            
    # L2 = torch.nn.MSELoss()(observation_bary1,observation_bary2)
    L2 = torch.linalg.norm((observation_bary1-observation_bary2), axis=-1)
    
    # print('Shape of W2', W2.shape)
    # print('Shape of L2', L2.shape)
    
    return (W2/L2).cpu().numpy()


def compute_sup_wts_Cs(measures, field_coordinates, params_opt, wts, sensor_locations):
    
    wts[0] = torch.nn.Parameter(wts[0])
    wts[1] = torch.nn.Parameter(wts[1])
    
    gamma = params_opt['gamma']
    niter = params_opt['nmax']
    momentum = params_opt['momentum']
    
    def get_optimizer(params):
        if params_opt['optimizer'] == 'Adam':
            return torch.optim.Adam(params, lr=params_opt['lr'], maximize=True)
        # else:
        #     return torch.optim.SGD(params, lr=params_opt['lr'], momentum=0.9, maximize=True)
        
        elif params_opt['optimizer'] == 'SGD':
            return torch.optim.SGD(params, lr=params_opt['lr'], nesterov=True,
                                   momentum=momentum, maximize=True)
    
    N = measures.shape[1]
    D = dimension(measures)        
    
    # sensor_locations = torch.tensor(x, device=device)                
    
    # params = wts
    # params = self.wts
    optimizer = get_optimizer(wts)
    scheduler = ExponentialLR(optimizer, gamma=0.99)
    
    # ker, _ = self.convolution_kernel_v2(self.field_coordinates, sensor_locations)
    
    evolution = {'loss':[],'weights1':[],'weights2':[], 'sup_stability_const':[]}
    
    for iter in tqdm(range(niter)):
        
        evolution['weights1'].append(wts[0].detach().clone())
        evolution['weights2'].append(wts[1].detach().clone())
        
        optimizer.zero_grad()
        
        f = partial(compute_Cs, measures, field_coordinates, sensor_locations=sensor_locations)
        
        loss, bary1, bary2, W2, _, _, _ = f(wts)
        
        loss.backward()
        optimizer.step()
        # scheduler.step()
                    
        wts[0].data= mat2simplex(wts[0]*gamma)
        wts[1].data= mat2simplex(wts[1]*gamma)
        
        evolution['loss'].append(loss.item())
        evolution['sup_stability_const'].append(loss.item())
            
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
    
    wts[0] = wts[0].detach().clone()
    wts[1] = wts[1].detach().clone()
    
    return wts, bary1.detach().clone(), bary2.detach().clone(), W2

# def compute_sup_wts_Cs(measures, field_coordinates, params_opt, wts, sensor_locations):
def compute_inf_x_Cs_SGD(measures, field_coordinates, params_opt, wts, sensor_locations):
    
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
    
    N = measures.shape[1]
    D = dimension(measures)
    field = field_coordinates.squeeze()
    
    #Initialization:
    # sensor_locations = torch.linspace(field[0].item()*0.9, field[-1].item()*0.9, no_of_sensors).tolist(); arange = 'even'
    # # sensor_locations = torch.linspace(0., 1., no_of_sensors).tolist(); arange = 'central'
    # sensor_locations = torch.nn.Parameter(torch.tensor([sensor_locations], dtype=dtype, device=device))        
    
    # params = wts
    # params = self.wts
    optimizer = get_optimizer(sensor_locations)
    scheduler = ExponentialLR(optimizer, gamma=0.99)
    
    # ker, _ = self.convolution_kernel_v2(self.field_coordinates, sensor_locations)
    
    evolution = {'loss':[],'weights1':[],'weights2':[], 'sup_stability_const':[]}
    
    for iter in tqdm(range(niter)):
        
        # evolution['weights1'].append(params[0].detach().clone())
        # evolution['weights2'].append(params[1].detach().clone())
        
        optimizer.zero_grad()
        # compute_Cs(measures, field_coordinates, params_opt, wts, sensor_locations)
        f = partial(compute_Cs, measures, field, wts)
        # W2/L2, bary1, bary2, W2.item(), L2, ker, cell_indices
        Cs, bary1, bary2, W2, L2, ker, cell_indices = f(sensor_locations)
        
        loss = L2
        
        loss.backward()
        optimizer.step()
        # scheduler.step()
        
        evolution['loss'].append(loss.item())
        evolution['sup_stability_const'].append(loss.item())
            
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
    
    print('sensor_locations', sensor_locations)
    
    return sensor_locations.detach().clone(), Cs, ker, cell_indices


def compute_inf_x_Cs_CBO(measures, field_coordinates, params_opt, wts, no_of_sensors):
    
    conf = {'alpha': 70.0,
        'dt': 0.05,
        # 'max_time': 5.0,
        'sigma': 10.0, #self.k*(3.0+self.variance),
        'lamda': 1.0,
        # 'batch_args':{
        # 'batch_size':200,
        # 'batch_partial': False},
        'd': no_of_sensors,
        'max_it': 10,
        'N': 10,
        'M': 2,
        'track_list': ['update_norm', 'energy','x', 'consensus', 'drift'],
        'resampling': True,
        'update_thresh': 0.002}
    
    x = cbx.utils.init_particles(shape=(conf['M'], conf['N'], conf['d']),
                                  x_min=field_coordinates[0].item()*0.9, x_max = field_coordinates[-1].item()*0.9)
    
    # Run the CBO algorithm
    it = 0
    t  = 0
    # C_s = np.zeros(config['max_it'])
    
    f = partial(compute_Cs_4_CBO, measures, field_coordinates, wts)
    
    dyn = CBO(f, x=x, f_dim='3D', noise='anisotropic', **conf)
                
    sched = scheduler(dyn, [multiply(name='alpha', factor=1.1, maximum=1e15),
                        #multiply(name='sigma', factor=1.005, maximum=6.)
                        ])
    
    while not dyn.terminate():                                    
        
        print('Infimum iteration #',it)
        print('Loss', dyn.f_min)        
        # print('x',dyn.x)
        print('best_particles', dyn.best_particle[0], dyn.best_particle[1])
        # inf_loss.append(dyn.f_min)
        it+=1
        
        dyn.step()
        sched.update()
        
    best_particle = torch.tensor(dyn.best_particle[0][None, :], device=device)
    Cs            = dyn.f_min[0]
    
    ker, cell_indices = convolution_kernel_v2(field_coordinates, best_particle)
    
    return best_particle, Cs, ker, cell_indices

def optimize_over_Cs(target: torch.Tensor, measures: torch.Tensor, 
                        field_coordinates: torch.Tensor,
                        prob_name,
                        params_opt = {'optimizer': 'SGD', 'lr': 0.001, 'nmax': 5,'type_loss':'W2', 'gamma':1, 'momentum': 0.5}):
    
    #Initialization:
    cycles = 2
    C_s    = np.zeros((3, cycles))
    
    N = measures.shape[1]
    D = dimension(measures)
    field = field_coordinates.squeeze()
    
    infimum_algo = 'CBO'
    
    frames_dir = check_create_dir(results_dir+'Compute_C_s_'+prob_name+'/'+'five_measures/{}/'.format(infimum_algo))
    
    for nfac in range(3):
        
        print('***** m = {}n'.format(nfac+1))
        
        #Initialization:
        no_of_sensors = (nfac+1)*N
        
        sensor_locations = torch.linspace(field[0].item()*0.9, field[-1].item()*0.9, no_of_sensors).tolist(); arange = 'even'
        # sensor_locations = torch.linspace(0., 1., no_of_sensors).tolist(); arange = 'central'
        # sensor_locations = torch.nn.Parameter(torch.tensor([sensor_locations], dtype=dtype, device=device))
        sensor_locations = torch.tensor([sensor_locations], dtype=dtype, device=device)
                
        l        = torch.linspace(-0.5, 0.5, N)
        gaussian = lambda x : torch.exp(-0.5*((x)/2)**2)/((2*(2*math.pi)**0.5))
        g        = gaussian(l)/gaussian(l).sum()

        weights1 = torch.ones((1,N), dtype=dtype, device=device)/N
        # weights2 = torch.nn.Parameter(torch.tensor([[0.5,0,0.5]], dtype=dtype, device=device))
        weights2 = torch.tensor([g.tolist()], dtype=dtype, device=device)
        
        wts      = [weights1, weights2]
    
        for i in range(cycles):
            
            # (measures, field_coordinates, params_opt, wts, sensor_locations):
            wts, bary1, bary2, W2 = compute_sup_wts_Cs(measures, field_coordinates, params_opt, wts, sensor_locations)
            
            # self.wts = wts
            
            # (measures, field_coordinates, params_opt, wts, sensor_locations):
            if infimum_algo == 'SGD':
                sensor_locations, Cs, ker, cell_indices = compute_inf_x_Cs_SGD(measures, field_coordinates, params_opt, wts, sensor_locations)
                
            else:
                pass
                sensor_locations, Cs, ker, cell_indices = compute_inf_x_Cs_CBO(measures, field_coordinates, params_opt, wts, no_of_sensors)
            
            if i == 0:
                sensors_history = sensor_locations.detach().cpu().clone().reshape(no_of_sensors,1)
            else:
                sensors_history = torch.concatenate((sensors_history,
                                                     sensor_locations.detach().cpu().clone().reshape(no_of_sensors,1)), axis=1)
            
            
            ######## Plotting frames begins ########
            fig, axs = plt.subplots(2, sharex=True, sharey=True)
            
            sensor_placement = torch.zeros(field_coordinates.shape[0])
            
            for j in range(measures.shape[1]): 
                axs[0].plot(field.cpu().numpy(), torch.squeeze(measures.cpu())[j].numpy())
            
            axs[0].plot(field.cpu().numpy(), sensor_placement.cpu().numpy(), markevery = cell_indices, ls = "", 
                      marker="o", markerfacecolor='blue', markersize=10)
            
            axs[0].grid(False)
            
            axs[0].set_xlabel('x')
            axs[0].set_title(r'Computing stability const. w/ {}ly dist. sensors'.format(arange))
            
            axs[1].plot(field.cpu().numpy(), bary1.cpu().detach().squeeze().numpy())
            axs[1].plot(field.cpu().numpy(), bary2.cpu().detach().squeeze().numpy())
            axs[1].plot(field.cpu().numpy(), sensor_placement.cpu().numpy(), markevery = cell_indices, ls = "", 
                      marker="o", markerfacecolor='blue', markersize=10)
            # axs[1].grid(False)
            axs[1].set_xlabel('x')
            axs[1].legend([r'Barycenter 1', 'Barycenter 2'],loc='best')
            # axs[1].title(r'$\inf_{x\in\Omega}\, \sup_{\Lambda_1,\Lambda_2} \frac{W_2(\mu,\nu)}{MSE(\mu,\nu)}$')
            
            axs2 = axs[1].twinx()
            
            for l in range(no_of_sensors): 
                tmp = ker[l][:].cpu().detach().numpy()
                axs2.plot(field_coordinates.squeeze().cpu().numpy(), tmp, 'blue', alpha = 0.1, 
                          linestyle='dotted')
                axs2.fill_between(field_coordinates.squeeze().cpu().numpy(), tmp, 0, color='blue', alpha=.03)
                    
            axs2.grid(False)
            
            fig.savefig(frames_dir+'frame_{}.png'.format(i))
            
            if i == 0 or i == int(cycles*0.25) or i == int(cycles*0.5) or i == cycles-1:
                fig.savefig(frames_dir+'Nsensors_{}n_frame_{}_{}.png'.format(nfac+1,i,arange))
            
            # plt.title(r"Hello \Tex", fontsize=6)
            plt.close()        
                
            # Save C_s values:
            print('nfac', nfac, 'i', i, 'C_s.shape', C_s.shape)
            C_s[nfac][i] = Cs
            
        df = pd.DataFrame(sensors_history.numpy())
        df.to_csv(frames_dir+'sensor_history_'+arange+'_'+str(no_of_sensors), index=False)
        print('Sensor locations written to file.')
        
        plot_sensor_trajectory(i, no_of_sensors, sensors_history, frames_dir,
                                   arange, cycles, prob_name)
        
        prob_name_gif = str('batch_')+prob_name+str('_{}_{}'.format(no_of_sensors,arange))
        
        plot_gif(i, frames_dir, prob_name_gif)
            
        Filename = frames_dir+'batch_C_s_'+str(no_of_sensors)+'_'+arange
        file = open(Filename+'.txt', "w")
        # for i in inf_evolution['inf_sup_stability_const']: file.write("%s\n" % i)
        for k in C_s[nfac]: file.write("%s\n" % k)
        file.close()
        print('Writing inf_loss to file complete')
        
        # if nfac == 0:
        #     C_s = np.array([inf_evolution['sup_stability_const']])
        # else:
        #     tmp = np.array([inf_evolution['sup_stability_const']])
        #     C_s = np.concatenate((C_s, tmp), axis=0)
        #     print('shape C_s', C_s.shape)
        
        torch.save(sensor_locations.detach().cpu().clone(), frames_dir+'{}_{}_sensor-coordinates'.format(no_of_sensors,arange))    
        # x = torch.load(frames_dir+'{}_{}_sensor-coordinates'.format(no_of_sensors,arange))
        # print(x)
        
    plot_graphs(cycles, nfac, C_s, frames_dir, prob_name, arange)
        
    # torch.save(best_particle[0], frames_dir+'{}_sensor-coordinates'.format(config['d']*(nfac+1)))
        
    # return sensors_history, C_s[None, :]

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