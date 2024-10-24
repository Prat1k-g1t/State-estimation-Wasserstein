# SPDX-FileCopyrightText: The sinkhorn-rom authors, see file AUTHORS
#
# SPDX-License-Identifier: GPL-3.0-or-later

from geomloss import SamplesLoss
# from sinkhorn_images import sinkhorn_divergence
import torch
import numpy as np
import argparse
import time
# from geomloss import SamplesLoss#ImagesBarycenter
from geomloss.sinkhorn_images import sinkhorn_divergence
from ..lib.DataManipulators.Problems import *
from ..lib.Evaluators.Barycenter import ImagesLoss, ImagesBarycenter_v2, \
    ImagesBarycenter_1d, projGraFixSupp, projGraAdapSupp
#from ..config import *
from ..config import results_dir, device, dtype
from ..visualization import plot_fields

import matplotlib.pyplot as plt
plt.rcParams.update(plt.rcParamsDefault)
plt.rcParams["font.family"] = 'serif'
plt.rcParams["savefig.format"] = 'pdf'
# plt.rcParams['text.usetex'] = True
import seaborn as sns
sns.set_theme(style="darkgrid")
sns.set(rc={"figure.dpi":100, 'savefig.dpi':100})
sns.set_context("paper")

#Parse
parser = argparse.ArgumentParser()
parser.add_argument('-n', type=int, default=100  , help='number of parameters')
parser.add_argument('-p', type=int, default=3,  help='number of snapshots to test')
args = parser.parse_args()

loss = SamplesLoss("sinkhorn", p=2, scaling = 0.9, debias=True)

# trial_measure1 = torch.zeros(100, device=device, dtype=dtype)
# trial_measure2 = torch.zeros(100, device=device, dtype=dtype)

# trial_measure1[0] = 2.; trial_measure2[99] = 2.
# trial_measure1 = trial_measure1/trial_measure1.sum()
# trial_measure2 = trial_measure2/trial_measure2.sum()
# trial_loss = ImagesLoss(trial_measure1[None, None, :], trial_measure2[None, None, :])
# print('W_2 dist. ImageLoss', (trial_loss*2)**0.5)
# trial_loss = loss(trial_measure1[None, None, :], trial_measure2[None, None, :])
# print('W_2 dist. Sinkhorn', trial_loss, trial_loss**0.5)


def debug_visualization(x,U):
#Used for debugging
    fig = plt.figure()
    for i in range(U.shape[1]):
        plt.plot(x.numpy(),U[0][i][:].numpy().T)
        plt.show()
    #plt.pause(0.1)
    # plt.close()

# Load data for problem
L = torch.randint(0, args.n, (args.p,))
# L = torch.tensor([9, 6, 9])
print('Random snapshots no. {} selected from a set of {} snapshots'.format(L,args.n))
problem_name = 'Gaussian1d'
problem = Problem(name = problem_name)
field_coordinates, snapshots, parameters, _ = problem.load_dataset(nparam=args.n)

# Let us compute Barycenter([a_1, \cdots, a_n], weights=[w_1, \cdots, w_n])
wts = torch.rand(args.p, dtype=dtype, device=device); wts /= torch.sum(wts)
weights_ref = wts
# weights_ref = torch.tensor([0.3207, 0.6570, 0.0223], dtype=dtype, device=device)
# weights_ref2 = torch.tensor([0.4996, 0.5004], dtype=dtype, device=device)
print('Random weights {} and its sum is {}'.format(weights_ref,torch.sum(weights_ref)))

measures_renorm = snapshots[0]/snapshots[0].sum()

measures_renorm = measures_renorm[None, None, :]
for i in range(args.p-1): 
    # tmp = snapshots[L[i+1].item()]/snapshots[L[i+1].item()].sum()
    # print('Mass of snapshot {}'.format(L[i+1]), 'is',torch.sum(tmp))
    tmp = snapshots[i+1]/snapshots[i+1].sum()
    print('Mass of snapshot {}'.format(i+1), 'is',torch.sum(tmp))
    measures_renorm = torch.cat((measures_renorm, tmp[None, None, :]), dim=1)

print('Shape of measures_renorm', measures_renorm.shape)

#DEBUG:
# print('Shape of snapshots[2]',snapshots[2].shape)
# print('Snapshots[2]',snapshots[2])
# print('Snapshots[10]',snapshots[10])
# print('Field coordinates', field_coordinates/8)
# d_samplesloss = loss(snapshots[2], field_coordinates/8, snapshots[10], field_coordinates/8) # loss(alpha, x_alpha, beta, y_beta) # (batch, N, dim)
# d_imagesloss  = ImagesLoss(snapshots[2][None, None, :], snapshots[10][None, None, :], scaling = 0.9, debias=True) # loss(alpha, x_alpha, beta, y_beta) # (batch, N, dim)
# d_sinkdiv     = 0. #sinkhorn_divergence(bestbary, barycenter_ref, scaling = 0.9) # loss(alpha, x_alpha, beta, y_beta) # (batch, N, dim)
# print('W2 distance is ', d_samplesloss, d_imagesloss, d_sinkdiv)


barycenter_ref = ImagesBarycenter_1d(measures=measures_renorm, weights=weights_ref[None,:], scaling_N = 200)
# barycenter_ref2 = ImagesBarycenter_1d(measures=measures_renorm, weights=weights_ref2[None,:], scaling_N = 300)

print('Mass of ref barycenter',torch.sum(barycenter_ref))

####################

# Compute barycenter 

params_sinkhorn_bary = {'blur': 0. ,'p':2, 'scaling_N':100,'backward_iterations':5} 
params_opt_best_barycenter = {'optimizer': 'Adam', 'nmax': 50, 'gamma': 1, 'k_sparse': 5}

bestbary, weights, _ = projGraAdapSupp(
                        target_field=barycenter_ref,
                        measures = measures_renorm,
                        field_coordinates=field_coordinates)
                        # params_opt = params_opt_best_barycenter,
                        # params_sinkhorn_bary=params_sinkhorn_bary,
                        # )

# Print output weights
print('Mass of best barycenter is',torch.sum(bestbary))
print('Reference weights are {}. Computed weights are {} with sum {}'.format(weights_ref, weights, weights.sum()))

####################
# Generating outputs

fig, axs = plt.subplots(2, sharex=True, sharey=True)

x = field_coordinates.cpu().squeeze().numpy()

for i in range(args.p): axs[0].plot(x, torch.squeeze(measures_renorm.cpu())[i].numpy())
axs[0].set_xlabel('x')
axs[0].set_ylabel(r'$\mu_{normalized}$', rotation=90)
axs[0].legend([r'$\mu_{},\lambda_{}={}$'.format(i+1,i+1,weights_ref[i]) for i in range(args.p)],loc='best')

axs[1].plot(x, barycenter_ref.cpu().detach().squeeze().numpy())
# axs[1].plot(x, barycenter_ref2.detach().squeeze().numpy())
axs[1].plot(x, bestbary.cpu().detach().squeeze().numpy())
axs[1].set_xlabel('x')
axs[1].legend([r'Ref. barycenter', 'best barycenter'],loc='best')

fig.savefig(results_dir+problem_name)

####################

# Error analysis between best barycenter and convex combination of weights and snapshots
    
# Err = bestbary.squeeze() - barycenter_ref.squeeze() # error_analysis(bestbary, weights_ref, snapshots, args.p)

# L1 = torch.norm(Err, p=1); L2 = torch.norm(Err, p=2); Linf = torch.norm(Err, float('inf'))
# print('Errors: L1 {}, L2 {}, Linf {}'.format(L1,L2,Linf))

# d_samplesloss = loss(bestbary.squeeze(),  field_coordinates/8, barycenter_ref.squeeze(),  field_coordinates/8) # loss(alpha, x_alpha, beta, y_beta) # (batch, N, dim)
d_imagesloss  = ImagesLoss(bestbary, barycenter_ref) # loss(alpha, x_alpha, beta, y_beta) # (batch, N, dim)
# d_sinkdiv     = 0. #sinkhorn_divergence(bestbary, barycenter_ref, scaling = 0.9) # loss(alpha, x_alpha, beta, y_beta) # (batch, N, dim)
print('W2 distance is ', d_imagesloss)


