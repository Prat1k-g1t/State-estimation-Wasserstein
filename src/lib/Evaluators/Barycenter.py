# SPDX-FileCopyrightText: The sinkhorn-rom authors, see file AUTHORS
#
# SPDX-License-Identifier: GPL-3.0-or-later

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

from ...config import results_dir, fit_dir, use_pykeops, device
from ...utils import check_create_dir

import matplotlib.pyplot as plt
import matplotlib

from scipy.linalg import toeplitz
import math
import imageio

import sys

from typing import Callable, Any



# def best_barycenter(target_field: torch.Tensor,
#                     fields: torch.tensor,
#                     field_coordinates: torch.Tensor,
#                     Loss,
#                     C: torch.Tensor = None,
#                     params_opt = {'optimizer': 'Adam', 'eps_conv': 1.e-6, 'nmax': 3},
#                     params_sinkhorn_bary = {'logsumexp': None, 'spatial_metric': 2, 'eps': 1.e-2, 'tau1': 10., 'tau2': 10., 'nmax': 100}
#                     ):

#     def get_optimizer(alpha):
#         if params_opt['optimizer'] == 'Adam':
#             return torch.optim.Adam([alpha], lr=1e-3)
#         elif params_opt['optimizer'] == 'LBFGS':
#             return torch.optim.LBFGS([alpha], max_iter=2)
#         elif params_opt['optimizer'] == 'SGD':
#             return torch.optim.SGD([alpha], lr=0.1, momentum=0.9)
#         elif params_opt['optimizer'] == 'Rprop':
#             return torch.optim.Rprop([alpha], lr=0.01, etas=(0.5, 1.2), step_sizes=(1e-06, 50))
#         elif params_opt['optimizer'] == 'RMSprop':
#             return torch.optim.RMSprop([alpha], lr=0.01, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0, centered=False)
#         elif params_opt['optimizer'] == 'ASGD':
#             return torch.optim.ASGD([alpha], lr=0.01, lambd=0.0001, alpha=0.75, t0=1000000.0, weight_decay=0)
#         elif params_opt['optimizer'] == 'Adadelta':
#             return torch.optim.Adadelta([alpha], lr=1.0, rho=0.9, eps=1e-06, weight_decay=0)
#         elif params_opt['optimizer'] == 'Adagrad':
#             return torch.optim.Adagrad([alpha], lr=0.01, lr_decay=0, weight_decay=0, initial_accumulator_value=0, eps=1e-10)
#         else:
#             print('Could not find optimizer {}. Running with Adam instead.'.format(params_opt['optimizer']))
#             return torch.optim.Adam([alpha], lr=1e-2)

#     # n = len(fields)
#     n = fields.shape[1]
#     alpha = torch.nn.Parameter(torch.ones(n, dtype=dtype, device=device)/n)
#     optimizer = get_optimizer(alpha)

#     def closure():
#         '''
#         The function `closure()` contains the same steps we typically use before taking a step with SGD or Adam. In other words, if the optimizer needs the gradient once, like SGD or Adam, it is simple to calculate the gradient with `.backward()` and pass it to the optimizer. If the optimizer needs to calculate the gradient itself, like LBFGS, then we pass instead a function that wraps the steps we typically do once for others optimizers. 
#         '''
#         optimizer.zero_grad()
#         weights = torch.exp(alpha)/torch.sum(torch.exp(alpha))
#         # bary = barycenter(fields = fields,weights=weights,field_coordinates=field_coordinates,**params_sinkhorn_bary)
#         bary = ImagesBarycenter_1d(fields = fields, weights=weights,**params_sinkhorn_bary)
#         loss = Loss(target_field, field_coordinates, bary, field_coordinates)
#         loss.backward(retain_graph=True)
#         closure.weights = weights
#         closure.bary = bary
#         return loss

#     torch.autograd.set_detect_anomaly(True)
#     # Optimization steps
#     pbar = tqdm(range(params_opt['nmax']))
#     for i in pbar:
#         optimizer.step(closure)
#     return closure.bary, closure.weights



def mat2simplex(vecX,l=1.0):
    n,m = vecX.shape
    vecS = torch.sort(vecX,dim=1,descending=True)[0]
    vecC = torch.cumsum(vecS,dim=1).to(device)-l
    vecH = vecS-vecC/(torch.arange(m)+1).reshape(1,m).to(device)
    vecH[vecH<=0]= np.inf
    r = torch.argmin(vecH,dim=1)
    t = vecC[torch.arange(n),r]/(r+1)
    vecY = vecX-t.reshape(-1,1)
    vecY[vecY<0]=0
    return vecY

def projGraFixSupp(target_field: torch.Tensor,
                    measures: torch.Tensor,
                    field_coordinates: torch.Tensor,
                    params_opt = {'optimizer': 'Adam', 'lr': 0.001, 'nmax': 100,'type_loss':'W2', 'gamma':1, 'k_sparse':5 },
                    params_sinkhorn_bary = {'blur': 0.001 , 'p':2, 'scaling_N':300,'backward_iterations':0},
                    distance_target_measures = None):
    N = measures.shape[1]
    weights = torch.nn.Parameter(torch.ones((1,N), dtype=dtype, device=device)/N)
    # eps = 0.
    # # A = np.exp(np.array([[0.5-eps,0.5+eps]]))
    # A = np.array([[0.5-eps,0.5+eps]])
    # weights = torch.nn.Parameter(torch.tensor(np.array([[0.5,0.5]]), dtype=dtype, device=device))
    # weights = torch.exp(weights)
    n_keep = params_opt['k_sparse']
    gamma = params_opt['gamma']
    niter = params_opt['nmax']
    
    eps = 1e-3
    Loss = SamplesLoss("sinkhorn", blur = eps, scaling=.9, debias=True)
    
    D = dimension(target_field)

    def get_optimizer(weights):
        if params_opt['optimizer'] == 'Adam':
            return torch.optim.Adam([weights], lr=params_opt['lr'])
        else:
            return torch.optim.SGD([weights], lr=params_opt['lr'], momentum=0.9)
        
    optimizer = get_optimizer(weights)
    
    # DEBUG:
    # print('Wts begin', weights)
    
    S = torch.arange(N,device=device)
    evolution = {'loss':[],'support':[],'weight':[], 'true_error':[]}

    if params_opt['type_loss']=='mse':
        total_measures = torch.cat([measures[0][id][None,None,:,:] for id in range(N)], dim=0)
        targets =  target_field.repeat(N, 1, 1, 1).to(dtype=dtype)
        if distance_target_measures is None:
            distance_true = ImagesLoss(targets,total_measures,scaling=0.9)
            # distance_true = Loss(targets.flatten(), field_coordinates, \
            #                      total_measures.flatten(), field_coordinates)
        else:
            distance_true = distance_target_measures.to(dtype=dtype)
    
    tmp_loss = 1e20; tols = 1e-12
    
    for iter in tqdm(range(niter)):
    # for iter in range(niter):
        evolution['weight'].append(weights.detach().clone())
        optimizer.zero_grad()
        if D == 1:
            # print('weights',weights)
            bary = ImagesBarycenter_1d(measures=measures, weights=weights)
            # print('Here')
        else:
            bary  = ImagesBarycenter_v2(measures=measures, weights=weights,**params_sinkhorn_bary)
        if params_opt['type_loss']=='mse':
            barycenters = bary.repeat(N,1,1,1)
            true_error = ImagesLoss(target_field, bary,blur=0.001,scaling=0.9)
            #distance_true = ImagesLoss(targets,total_measures,blur=0.001,scaling=0.8)
            distance_approx = ImagesLoss(barycenters,total_measures,blur=0.001,scaling=0.8)
            loss = N*torch.nn.MSELoss()(distance_true, distance_approx)
            evolution['true_error'].append(true_error.item())

        else:
            if D == 1:
                loss = ImagesLoss(target_field, bary, scaling=0.9)
                # loss = Loss(target_field, bary)
                # print('loss interior',loss)
            else:
                loss = ImagesLoss(target_field, bary, blur=0.0001,scaling=0.9)
            evolution['true_error'].append(loss.item())
        # loss = loss_1 
        loss.backward()
        optimizer.step()
        
        #DEBUG:
        # print('loss.grad',loss.grad)
        # print('weights.grad',weights.grad)
        # print('Requires grad: target_field, measures, bary, weights',
        #       target_field.requires_grad, measures.requires_grad, bary.requires_grad, weights.requires_grad)
        
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
            # weights.data= weights/weights.sum()
        S_index= torch.nonzero(torch.where(weights > 1.e-4, weights, 0.).flatten()).flatten()
        #print('Iter {}, weights {}, loss = {}'.format(iter,weights,loss.item()))
    
        #DEBUG:
        # print('Wts after', weights)
        
        #print('supp', S[S_index])
        evolution['loss'].append(loss.item())
        evolution['support'].append( S[S_index].cpu())
        
        # if tmp_loss - loss.item() > tols:
        #     tmp_loss = loss.item()
        #     # print('Modify to add tols to the grad of the loss. Remove print when implemented')
        # else:
        #     print('Reached grad loss under tolerance ({})'.format(tols))
        #     break
        
    # print('iter', iter, 'loss', loss)
    return bary, weights, evolution


def projGraAdapSupp(target_field: torch.Tensor,
                    measures: torch.Tensor,
                    field_coordinates: torch.Tensor,
                    params_opt = {'optimizer': 'SGD', 'lr': 0.001,'type_loss':'W2', 'nmax': 100, 'gamma':1},
                    params_sinkhorn_bary = {'blur': 0.001 , 'p':2, 'scaling_N':100,'backward_iterations':7},
                    distance_target_measures = None):
    eps = 1e-3
    Loss = SamplesLoss("sinkhorn", blur = eps, scaling=.9, debias=True)
    
    N = measures.shape[1]
    weights = torch.nn.Parameter(torch.ones((1,N), dtype=dtype, device=device)/N)
    
    gamma = params_opt['gamma']
    niter = params_opt['nmax']
    
    D = dimension(target_field)    

    def get_optimizer(weights):
        if params_opt['optimizer'] == 'Adam':
            return torch.optim.Adam([weights], lr=params_opt['lr'])
        else:
            return torch.optim.SGD([weights], lr=params_opt['lr'], momentum=0.9)
        
    optimizer = get_optimizer(weights)
    
    S = torch.arange(N,device=device)
    evolution = {'loss':[],'support':[],'weight':[], 'true_error':[]}
    if params_opt['type_loss']=='mse':
        total_measures = torch.cat([measures[0][id][None,None,:,:] for id in range(N)], dim=0)
        targets =  target_field.repeat(N, 1, 1, 1).to(dtype=dtype)

        if distance_target_measures is None:
            distance_true = ImagesLoss(targets,total_measures,blur=0.001,scaling=0.8)
        else:
            distance_true = distance_target_measures.to(dtype=dtype)
            #print('Use approximation for distance_target_measures!')
    
    tmp_loss = 1e20; tols = 1e-6
    # Initial value for n_keep:
    n_keep = N
    for iter in tqdm(range(niter)):
        evolution['weight'].append(weights.detach().clone())
        optimizer.zero_grad()
        if D == 1:
            bary  = ImagesBarycenter_1d(measures=measures, weights=weights)
        else:
            bary  = ImagesBarycenter_v2(measures=measures, weights=weights,**params_sinkhorn_bary)
        

        if params_opt['type_loss']=='mse':
            #print('MSE')
            barycenters = bary.repeat(N,1,1,1)
            true_error = ImagesLoss(target_field, bary,blur=0.0001,scaling=0.9)
            #distance_true = ImagesLoss(targets,total_measures,blur=0.001,scaling=0.8)
            distance_approx = ImagesLoss(barycenters,total_measures,blur=0.001,scaling=0.8)
            loss = N*torch.nn.MSELoss()(distance_true, distance_approx)
            evolution['true_error'].append(true_error.item())

        else:
            if D == 1:
                loss = ImagesLoss(target_field, bary,scaling=0.9)
            else:
                loss = ImagesLoss(target_field, bary,blur=0.0001,scaling=0.9)
            # loss = Loss(target_field, bary)
            evolution['true_error'].append(loss.item())
            #print('W2')

        loss.backward()
        optimizer.step()
        
        # keep n largest value 
        if n_keep < N:
            index_sort= torch.argsort(weights.data,dim=1,descending=True).flatten()
            index_keep = index_sort[:n_keep]
            index_zero = index_sort[n_keep:]
            weights.data[:,index_zero]=0
            
            # projection on the simplex
            weights.data[:,index_keep] = mat2simplex(weights[:,index_keep]*gamma)
        else:
            weights.data= mat2simplex(weights*gamma)
        S_index= torch.nonzero(torch.where(weights > 1.e-4, weights, 0.).flatten()).flatten()
        n_keep = S_index.shape[0]
        #print('Iter {}, weights {}, loss = {}, true_error = {}'.format(iter,weights,loss.item(),true_error.item()))
        #print('supp', S[S_index])
        evolution['loss'].append(loss.item())
        evolution['support'].append( S[S_index].cpu())
        
        # if tmp_loss - loss.item() > tols:
        #     tmp_loss = loss.item()
        # else:
        #     # print('Reached grad loss under tolerance ({})'.format(tols))
        #     break
        

    return bary, weights, evolution

def projRGraSP(target_field: torch.Tensor,
                    measures: torch.Tensor,
                    field_coordinates: torch.Tensor,
                    params_opt = {'optimizer': 'Adam', 'lr': 0.01, 'nmax': 100, 'type_loss':'W2', 'gamma':1, 'k_sparse':5 },
                    params_sinkhorn_bary = {'blur': 0.001 , 'p':2, 'scaling_N':100,'backward_iterations':7},
                    distance_target_measures = None):
    N = measures.shape[1]
    weights = torch.nn.Parameter(torch.ones((1,N), dtype=dtype, device=device)/N)
    
    n_keep = params_opt['k_sparse']
    gamma = params_opt['gamma']
    niter = params_opt['nmax']

    def get_optimizer(weights):
        if params_opt['optimizer'] == 'Adam':
            return torch.optim.Adam([weights], lr=params_opt['lr'])
        else:
            return torch.optim.SGD([weights], lr=params_opt['lr'], momentum=0.9)
        
    optimizer = get_optimizer(weights)
    
    S = torch.arange(N,device=device)
    evolution = {'loss':[],'support':[],'weight':[]}
    # prepare for mse loss
    if params_opt['type_loss']=='mse':
        total_measures = torch.cat([measures[0][id][None,None,:,:] for id in range(N)], dim=0)
        targets =  target_field.repeat(N, 1, 1, 1).to(dtype=dtype)
        if distance_target_measures is None:
            distance_true = ImagesLoss(targets,total_measures,blur=0.001,scaling=0.8)
        else:
            distance_true = distance_target_measures.to(dtype=dtype)
        

    for iter in tqdm(range(niter)):
        evolution['weight'].append(weights.detach().clone())
        optimizer.zero_grad()
        bary  = ImagesBarycenter_v2(measures=measures, weights=weights,**params_sinkhorn_bary)

        if params_opt['type_loss']=='mse':
            barycenters = bary.repeat(N,1,1,1)
            #true_error = ImagesLoss(target_field, bary,blur=0.0001,scaling=0.9)
            #distance_true = ImagesLoss(targets,total_measures,blur=0.001,scaling=0.8)
            distance_approx = ImagesLoss(barycenters,total_measures,blur=0.001,scaling=0.8)
            loss = N*torch.nn.MSELoss()(distance_true, distance_approx)

        else:
            loss = ImagesLoss(target_field, bary,blur=0.0001,scaling=0.9)

        loss.backward()


        with torch.no_grad():
            z_index_sort= torch.argsort(weights.grad,dim=1,descending=True).flatten()
            Z  = z_index_sort[:2*n_keep]
            w_index_sort= torch.argsort(weights.data,dim=1,descending=True).flatten()
            W  = w_index_sort[:n_keep]
            T = torch.unique(torch.cat((Z,W)))
            
        optimizer.step()

        # keep n largest value on T support
        if n_keep < T.shape[0]:
            index_sort_T= torch.argsort(weights.data[:,T],dim=1,descending=True).flatten()
            index_keep_T = index_sort_T[:n_keep]
            index_zero_T = index_sort_T[n_keep:]
            
            weights.data[:,T[index_zero_T]]=0
            # projection on the simplex
            weights.data[:,T[index_keep_T]] = mat2simplex(weights[:,T[index_keep_T]]*gamma)
        else:
            weights.data[:,T]= mat2simplex(weights[:,T]*gamma)

        # set zeros on S/T
        combined = torch.cat((S, T))
        uniques, counts = combined.unique(return_counts=True)
        difference = uniques[counts == 1]
        #print('D',difference)
        weights.data[:,difference] = 0.
        
        S_index= torch.nonzero(torch.where(weights > 1.e-4, weights, 0.).flatten()).flatten()
        #print('Iter {}, weights {}, loss = {}'.format(iter,weights,loss.item()))
        #print('supp', S[S_index])
        evolution['loss'].append(loss.item())
        evolution['support'].append( S[S_index].cpu())

    return bary, weights, evolution

def barycenter_iteration(f_k, g_k, d_log, eps, p, ak_log, w_k):

    # Sinkhorn "pseudo-step" - from the measures to the barycenter:
    ft_k = softmin(eps, p, ak_log + g_k / eps) / eps  # (B,K,n,n)
    # Update the barycenter:
    # (B,1,n,n) = (B,1,n,n) - (B,K,n,n) @ (B,K,1,1)
    bar_log = d_log - (ft_k * w_k[:, :, None, None]).sum(1, keepdim=True)

    # Symmetric Sinkhorn updates:
    # From the measures to the barycenter:
    #rho = 0.004
    #damping = rho / (rho + eps )

    ft_k = softmin(eps, p, ak_log + g_k / eps)  # (B,K,n,n)
    # From the barycenter to the measures:
    gt_k = softmin(eps, p, bar_log + f_k / eps)  # (B,K,n,n)
    f_k = (f_k + ft_k) / 2
    g_k = (g_k + gt_k) / 2

    # Sinkhorn "pseudo-step" - from the measures to the barycenter:
    ft_k = softmin(eps, p, ak_log + g_k / eps) / eps
    # Update the barycenter:
    # (B,1,n,n) = (B,1,n,n) - (B,K,n,n) @ (B,K,1,1)
    bar_log = d_log - (ft_k * w_k[:, :, None, None]).sum(1, keepdim=True)
    bar_log = bar_log- bar_log.logsumexp([2, 3], keepdim=True)
    
    # Update the de-biasing measure:
    # (B,1,n,n) = (B,1,n,n) + (B,1,n,n) + (B,1,n,n)
    d_log = 0.5 * (d_log + bar_log + softmin(eps, p, d_log) / eps)

    return f_k, g_k, d_log, bar_log


def ImagesBarycenter_v2(measures, weights, blur=0, p=2, scaling_N=200, backward_iterations=5):

    a_k = measures  # Densities, (B,K,N,N)
    w_k = weights  # Barycentric weights, (B,K)

    # Default precision settings: blur = pixel size.
    if blur == 0:
        blur = 1 / measures.shape[-1]

    with torch.set_grad_enabled(backward_iterations == 0):
        #print('MS')
        # Initialize the barycenter as a pointwise linear combination:
        bar = (a_k * w_k[:, :, None, None]).sum(1)  # (B,K,N,N) @ (B,K,1,1) -> (B,N,N)

        # Pre-compute a multiscale decomposition (=QuadTree)
        # of the input measures, stored as logarithms
        # List of 1X1, 2X2, 4X4, ... etc. domains
        ak_s = pyramid(a_k)[1:]  # We remove the 1x1 image, keep the 2x2, 4x4...
        ak_log_s = list(map(log_dens, ak_s))  # The code below relies on log-sum-exps

        # Initialize the blur scale at 1, i.e. the full image length:
        sigma = 1  # sigma = blur scale
        eps = sigma ** p  # eps = temperature

        # Initialize the dual variables
        f_k, g_k = softmin(eps, p, ak_log_s[0]), softmin(eps, p, ak_log_s[0])

        # Logarithm of the debiasing term:
        d_log = torch.ones_like(ak_log_s[0]).sum(dim=1, keepdim=True)  # (B,1,2,2)
        #DEBUG:
        # print('Shape of d_log',d_log.shape)
        d_log = d_log - d_log.logsumexp([2, 3], keepdim=True)  # Normalize each 2x2 image

        # Multiscale descent, with eps-scaling:
        # We iterate over sub-sampled images of shape nxn = 2x2, 4x4, ..., NxN
        for n, ak_log in enumerate(ak_log_s):
            for _ in range(scaling_N):  # Number of steps per scale
                # Update the temperature:
                eps = sigma ** p

                f_k, g_k, d_log, bar_log = barycenter_iteration(
                    f_k, g_k, d_log, eps, p, ak_log, w_k
                )

                # Decrease the kernel radius, making sure that
                # sigma is divided by two at every scale until we reach
                # the target value, "blur":
                sigma = max(sigma * (2 ** (-1 / scaling_N)), blur)

            if n + 1 < len(ak_s):  # Re-fine the maps, if needed
                f_k = upsample(f_k)
                g_k = upsample(g_k)
                d_log = upsample(d_log)

    if (measures.requires_grad or weights.requires_grad) and backward_iterations > 0:
        #print('BK')
        for _ in range(backward_iterations):
            f_k, g_k, d_log, bar_log = barycenter_iteration(
                f_k, g_k, d_log, eps, p, ak_log, w_k
            )

    return bar_log.exp()

def barycenter_iteration_1d(f_k, g_k, d_log, eps, p, ak_log, w_k):

    # Sinkhorn "pseudo-step" - from the measures to the barycenter:
    ft_k = softmin(eps, p, ak_log + g_k / eps) / eps  # (B,K,n,n)
    # Update the barycenter:
    # (B,1,n,n) = (B,1,n,n) - (B,K,n,n) @ (B,K,1,1)
    bar_log = d_log - (ft_k * w_k[:, :, None]).sum(1, keepdim=True)

    # Symmetric Sinkhorn updates:
    # From the measures to the barycenter:
    #rho = 0.004
    #damping = rho / (rho + eps )

    ft_k = softmin(eps, p, ak_log + g_k / eps)  # (B,K,n,n)
    # From the barycenter to the measures:
    gt_k = softmin(eps, p, bar_log + f_k / eps)  # (B,K,n,n)
    f_k = (f_k + ft_k) / 2
    g_k = (g_k + gt_k) / 2

    # Sinkhorn "pseudo-step" - from the measures to the barycenter:
    ft_k = softmin(eps, p, ak_log + g_k / eps) / eps
    # Update the barycenter:
    # (B,1,n,n) = (B,1,n,n) - (B,K,n,n) @ (B,K,1,1)
    bar_log = d_log - (ft_k * w_k[:, :, None]).sum(1, keepdim=True)
    bar_log = bar_log- bar_log.logsumexp([2], keepdim=True)
    
    # Update the de-biasing measure:
    # (B,1,n,n) = (B,1,n,n) + (B,1,n,n) + (B,1,n,n)
    d_log = 0.5 * (d_log + bar_log + softmin(eps, p, d_log) / eps)

    return f_k, g_k, d_log, bar_log

def ImagesBarycenter_1d(measures, weights, blur=0., p=2, scaling_N=100, backward_iterations=5):

    # Ongoing version of Sinkhorn barycenter for 1d measures.
    # (B,K,N): ({0 to no. of batch, where batches are collection of meas.},
                # K = No. of barycenter functions, \mu_1..\mu_n),i.field[0][None,:,:]
                # N = No. of spatial points, e.g. 100)
                
    # (B,K): (0 to no. of batch, where batches are collection of wts.},
              # K = set of wts)
    a_k = measures  # Densities, (B,K,N)
    w_k = weights  # Barycentric weights, (B,K)
    
    # visualization(x, a_k)

    # Default precision settings: blur = pixel size.
    if blur == 0:
        blur = 2 / measures.shape[-1]
        
    # print('blur',blur)

    with torch.set_grad_enabled(backward_iterations == 0):
        #print('MS')
        # Initialize the barycenter as a pointwise linear combination:
        bar = (a_k * w_k[:, :, None]).sum(1)  # (B,K,N) @ (B,K,1) -> (B,N)

        # Pre-compute a multiscale decomposition (=QuadTree)
        # of the input measures, stored as logarithms
        # --> Should be adapted to 1D
        
        #TODO: check average-pooling for 1D implementation
        ak_s = [a_k]  # We remove the 1x1 image, keep the 2x2, 4x4...
        ak_log_s = list(map(log_dens, ak_s))  # The code below relies on log-sum-exps
        #DEBUG:
        # print('len of ak_log_s',len(ak_log_s),'shape of ak_log_s[0]',ak_log_s[0].shape)

        # Initialize the blur scale at 1, i.e. the full image length:
        sigma = 1  # sigma = blur scale
        blur_list = np.geomspace(sigma, blur, scaling_N)
        eps = blur_list[0] ** p  # eps = temperature

        # Initialize the dual variables
        #TODO: Check 1D implementation in geomloss
        # print('log_dens(a_k).shape', log_dens(a_k).shape)
        # f_k, g_k = softmin(eps, p, log_dens(a_k)), softmin(eps, p, log_dens(a_k))  # Needs to be adapted to 1D
        f_k, g_k = softmin(eps, p, ak_log_s[0]), softmin(eps, p, ak_log_s[0])  # Needs to be adapted to 1D

        # Logarithm of the debiasing term:
        #TODO: check for 1D implementation
        d_log = torch.ones_like(ak_log_s[0]).sum(dim=1, keepdim=True)  # (B,1,2,2)
        # print('d_log',d_log)
        d_log = d_log - d_log.logsumexp([2], keepdim=True)  # Normalize each 2x2 image

        # Multiscale descent, with eps-scaling:
        # We iterate over sub-sampled images of shape nxn = 2x2, 4x4, ..., NxN
        for n, ak_log in enumerate(ak_log_s): #Siggma will go from 1 to blur
            for i in range(scaling_N):  # Number of steps per scale
        #         # Update the temperature:
                eps = blur_list[i] ** p
                # print('blur_list[i],eps_updated',blur_list[i],eps)

        #         ak_log = log_dens(a_k)
                f_k, g_k, d_log, bar_log = barycenter_iteration_1d(
                    f_k, g_k, d_log, eps, p, ak_log, w_k
                )

        #         # Decrease the kernel radius, making sure that
        #         # sigma is divided by two at every scale until we reach
        #         # the target value, "blur":
                # sigma = max(sigma * (2 ** (-1 / scaling_N)), blur)

            # if n + 1 < len(ak_s):  # Re-fine the maps, if needed
            #     f_k = upsample(f_k)
            #     g_k = upsample(g_k)
            #     d_log = upsample(d_log)
            
        #Avoiding average-pooling for 1D
        # ak_log = log_dens(a_k)
        # print('Shape of d_log and ak_log',d_log.shape,ak_log.shape)
        # f_k, g_k, d_log, bar_log = barycenter_iteration_1d(
        #     f_k, g_k, d_log, eps, p, ak_log_s[0], w_k)

    if (measures.requires_grad or weights.requires_grad) and backward_iterations > 0:
        #print('BK')
        for _ in range(backward_iterations):
            f_k, g_k, d_log, bar_log = barycenter_iteration_1d(
                    f_k, g_k, d_log, eps, p, ak_log_s[0], w_k
            )

    return bar_log.exp()

def visualization(x,U): #U(None, None, :)
    #DEBUG:
    # print('Nonzero U', torch.nonzero(U))
    
    fig = plt.figure()
    for i in range(U.shape[1]):
        plt.stem(x.numpy(),U[0][i][:].numpy())
        plt.show()
    #plt.pause(0.1)
    # plt.close()


def ImagesBarycenter_mass(fields, weights, blur=0.001, p=2, scaling_N = 50):

    if weights.dim() == 1:
        weights = weights.reshape(1,-1)

    measures = torch.cat([field[None,None,:,:] for field in fields], dim=1)

    a_k = measures  # (B, K, N, N)
    w_k = weights   # (B,K)
    if blur == 0:
        blur = 1 / measures.shape[-1]
    # normalization
    bar = (a_k * w_k[:,:,None,None]).sum(dim=1,keepdim=True)
    bar_mass = bar.sum(dim=[2, 3],keepdims=True)
    mass = a_k.sum(dim=[2,3],keepdim=True)
    a_k = a_k*(1./mass)


    with torch.enable_grad() :
        # Pre-compute a multiscale decomposition (=QuadTree)
        # of the input measures, stored as logarithms
        ak_s = pyramid(a_k)[1:]
        ak_log_s = list(map(log_dens, ak_s))

        # Initialize the dual variables
        sigma = 1 ; eps = sigma ** p  # sigma = blurring scale, eps = temperature
        f_k, g_k = softmin(eps, p, ak_log_s[0]), softmin(eps, p, ak_log_s[0])
        
        d_log = torch.ones_like(ak_log_s[0]).sum(dim=1, keepdim=True)
        d_log -= d_log.logsumexp([2, 3], keepdim=True)
        
        # Multiscale descent, with eps-scaling:
        for n, ak_log in enumerate( ak_log_s ):
            for _ in range(scaling_N) : # Number of steps per scale
                eps = sigma ** p

                # Update the barycenter:
                ft_k = softmin(eps, p, ak_log  + g_k / eps) / eps
                bar_log = d_log - (ft_k * w_k[:,:,None,None]).sum(1, keepdim=True)

                # symmetric Sinkhorn updates:
                ft_k = softmin(eps, p, ak_log  + g_k / eps)
                gt_k = softmin(eps, p, bar_log + f_k / eps)
                f_k += ft_k ; f_k *= .5 ; g_k += gt_k ; g_k *= .5

                # Update the barycenter:
                ft_k = softmin(eps, p, ak_log  + g_k / eps) / eps
                bar_log = d_log - (ft_k * w_k[:,:,None,None]).sum(1, keepdim=True)

                # Update the de-biasing measure:
                d_log = .5 * (d_log + bar_log + softmin(eps, p, d_log) / eps)

                # Decrease the kernel radius, making sure that
                # σ is divided by two at every scale until we reach
                # the target value, "blur":
                sigma = max(sigma * 2**(-1/scaling_N), blur)

                bar_log = bar_log- bar_log.logsumexp([2, 3], keepdim=True)
            #bar_log -= bar_log.logsumexp([2, 3], keepdim=True)
            if n+1 < len(ak_s):  # Re-fine the maps, if needed
                f_k = upsample(f_k) ; g_k = upsample(g_k)
                d_log = upsample(d_log)
    return bar_log.exp()*bar_mass

# def best_ImagesBarycenter(target_field: torch.Tensor,
#                     fields: List[torch.Tensor],
#                     measures: torch.Tensor,
#                     field_coordinates: torch.Tensor,
#                     Loss,
#                     params_opt = {'optimizer': 'LBFGS', 'eps_conv': 1.e-6, 'nmax': 10},
#                     params_sinkhorn_bary = {'blur': 0.001, 'p':2, 'scaling_N':50}, ):
#                     # params_sinkhorn_bary = {'blur': 0.001 , 'p':2, 'scaling_N':100,'backward_iterations':7}):

#     #print(params_sinkhorn_bary)
#     def get_optimizer(alpha):
#             print('Could not find optimizer {}. Running with Adam instead.'.format(params_opt['optimizer']))
#             return torch.optim.Adam([alpha], lr=1e-2)

#     n = len(fields)
#     n = measures.shape[1]
#     alpha = torch.nn.Parameter(torch.ones((1,n), dtype=dtype, device=device)/n)
#     optimizer = get_optimizer(alpha)

#         #weights = torch.exp(alpha)/torch.sum(torch.exp(alpha))
#         weights = torch.softmax(alpha, dim=1)
#         #images = torch.cat([field[None,None,:,:] for field in fields], dim=1)
        
#         bary  = ImagesBarycenterWithGrad(fields=fields, weights=weights,**params_sinkhorn_bary)
#         bary  = ImagesBarycenter_v2(measures=measures, weights=weights,**params_sinkhorn_bary)
#         #loss = Loss(target_field.flatten(), field_coordinates, bary.flatten(), field_coordinates)
#         loss = ImagesLoss(target_field, bary,blur=0.0001,scaling=0.9)
#         loss.backward(retain_graph=False) # check
#         optimizer.step(closure)
#     return closure.bary, closure.weights



#CLEAN:
# myDataset = PairWiseDataset(target)
# Total_loss = []
# firstIndex =[]
# secondIndex=[]
# train_dataloader = DataLoader(myDataset, batch_size=100, shuffle=False)
# for i_batch, sample_batched in enumerate(train_dataloader):
#     train_features, train_labels = sample_batched
#     loss = ImagesLoss(train_features[1], train_labels[1],blur=0.0001,scaling=0.9)
#     Total_loss.append(loss)
#     firstIndex.append(train_features[0])
#     secondIndex.append(train_labels[0])
    
def ImagesLoss(
    a,
    b,
    p=2,
    blur=None,
    reach=None,
    axes=None,
    scaling=0.5,
    cost=None,
    debias=True,
    potentials=False,
    **kwargs,
):
    r"""Sinkhorn divergence between measures supported on 1D/2D/3D grids.

    Args:
        a ((B, Nx), (B, Nx, Ny) or (B, Nx, Ny, Nz) Tensor): Weights :math:`\alpha_i`
            for the first measure, with a batch dimension.

        b ((B, Nx), (B, Nx, Ny) or (B, Nx, Ny, Nz) Tensor): Weights :math:`\beta_j`
            for the second measure, with a batch dimension.

        p (int, optional): Exponent of the ground cost function
            :math:`C(x_i,y_j)`, which is equal to
            :math:`\tfrac{1}{p}\|x_i-y_j\|^p` if it is not provided
            explicitly through the `cost` optional argument.
            Defaults to 2.

        blur (float or None, optional): Target value for the blurring scale
            of the "point spread function" or Gibbs kernel
            :math:`K_{i,j} = \exp(-C(x_i,y_j)/\varepsilon) = \exp(-\|x_i-y_j\|^p / p \text{blur}^p).
            In the Sinkhorn algorithm, the temperature :math:`\varepsilon`
            is computed as :math:`\text{blur}^p`.
            Defaults to None: we pick the smallest pixel size across
            the Nx, Ny and Nz dimensions (if applicable).

        [Important] axes (tuple of pairs of floats or None (= [0, 1)^(1/2/3)), optional):
            Dimensions of the image domain, specified through a 1/2/3-uple
            of [vmin, vmax] bounds.
            For instance, if the batched 2D images correspond to sampled
            measures on [-10, 10) x [-3, 5), you may use "axes = ([-10, 10], [-3, 5])".
            The (implicit) pixel coordinates are computed using a "torch.linspace(...)"
            across each dimension: along any given axis, the spacing between two pixels
            is equal to "(vmax - vmin) / npixels".

            Defaults to None: we assume that the signal / image / volume
            is sampled on the unit interval [0, 1) / square [0, 1)^2 / cube [0, 1)^3.

        scaling (float in (0, 1), optional): Ratio between two successive
            values of the blur radius in the epsilon-scaling annealing descent.
            Defaults to 0.5.

        cost (function or None, optional): ...
            Defaults to None: we use a Euclidean cost
            :math:`C(x_i,y_j) = \tfrac{1}{p}\|x_i-y_j\|^p`.

        debias (bool, optional): Should we used the "de-biased" Sinkhorn divergence
            :math:`\text{S}_{\varepsilon, \rho}(\al,\be)` instead
            of the "raw" entropic OT cost
            :math:`\text{OT}_{\varepsilon, \rho}(\al,\be)`?
            This slows down the OT solver but guarantees that our approximation
            of the Wasserstein distance will be positive and definite
            - up to convergence of the Sinkhorn loop.
            For a detailed discussion of the influence of this parameter,
            see e.g. Fig. 3.21 in Jean Feydy's PhD thesis.
            Defaults to True.

        potentials (bool, optional): Should we return the optimal dual potentials
            instead of the cost value?
            Defaults to False.

    Returns:
        (B,) Tensor or pair of (B, Nx, ...), (B, Nx, ...) Tensors: If `potentials` is True,
            we return a pair of (B, Nx, ...), (B, Nx, ...) Tensors that encode the optimal
            dual vectors, respectively supported by :math:`x_i` and :math:`y_j`.
            Otherwise, we return a (B,) Tensor of values for the Sinkhorn divergence.
    """

    D = dimension(a)
    
    if blur is None:
        if D == 1:
            blur = 2 / a.shape[-1]
        else:
            blur = 1 / a.shape[-1]

    # Pre-compute a multiscale decomposition (=Binary/Quad/OcTree)
    # of the input measures, stored as logarithms
    if D == 1:
        a_s, b_s = [a], [b]
        a_logs = list(map(log_dens, a_s))
        b_logs = list(map(log_dens, b_s))
    else:
        a_s, b_s = pyramid(a)[1:], pyramid(b)[1:]
        a_logs = list(map(log_dens, a_s))
        b_logs = list(map(log_dens, b_s))
        
    #DEBUG:
    # print('Len of a_logs', len(a_logs))
    # print('Shape of a_logs[0]', a_logs[0].shape)
    # print('Shape of a_logs[-1]', a_logs[-1].shape)

    # By default, our cost function :math:`C(x_i,y_j)` is a halved,
    # squared Euclidean distance (p=2) or a simple Euclidean distance (p=1):
    depth = len(a_logs)
    if cost is None:
        C_s = [p] * depth  # Dummy "cost matrices"
    else:
        raise NotImplementedError()

    # Diameter of the configuration:
    diameter = 1
    # Target temperature epsilon:
    eps = blur ** p
    # Strength of the marginal constraints:
    rho = None if reach is None else reach ** p

    # Schedule for the multiscale descent, with ε-scaling:
    """
    sigma = diameter
    for n in range(depth):
        for _ in range(scaling_N):  # Number of steps per scale
            eps_list.append(sigma ** p)

            # Decrease the kernel radius, making sure that
            # the radius sigma is divided by two at every scale until we reach
            # the target value, "blur":
            scale = max(sigma * (2 ** (-1 / scaling_N)), blur)

    jumps = [scaling_N * (i + 1) - 1 for i in range(depth - 1)]
    """
    if scaling < 0.5:
        raise ValueError(
            f"Scaling value of {scaling} is too small: please use a number in [0.5, 1)."
        )

    diameter, eps, eps_list, rho = scaling_parameters(
        None, None, p, blur, reach, diameter, scaling
    )
    #print('as',a_s[1].shape,'bs',b_s[1].shape)
    # print("eps_list: ", eps_list)
    # List of pixel widths:
    pyramid_scales = [diameter / a.shape[-1] for a in a_s]
    # print("Pyramid scales:", pyramid_scales)

    current_scale = pyramid_scales.pop(0)
    jumps = []
    for i, eps in enumerate(eps_list[1:]): #DEBUG: changes to 0: from 1:
        # if current_scale ** p > eps: 
        if current_scale ** p > eps and len(pyramid_scales)>=1:
            jumps.append(i+1)
    #         #print('eps',eps,'jum',jumps)
            current_scale = pyramid_scales.pop(0)

    # #print("Temperatures: ", eps_list)
    # print("Jumps: ", jumps)
    assert (
        len(jumps) == len(a_s) - 1
    ), "There's a bug in the multicale pre-processing..."

    # Use an optimal transport solver to retrieve the dual potentials:
    f_aa, g_bb, g_ab, f_ba = sinkhorn_loop(
        softmin_grid,
        a_logs,
        b_logs,
        C_s,
        C_s,
        C_s,
        C_s,
        eps_list,
        rho,
        jumps=jumps,
        kernel_truncation=kernel_truncation,
        extrapolate=extrapolate,
        debias=debias,
    )

    # Optimal transport cost:
    return sinkhorn_cost(
        eps,
        rho,
        a,
        b,
        f_aa,
        g_bb,
        g_ab,
        f_ba,
        batch=True,
        debias=debias,
        potentials=potentials,
    )
