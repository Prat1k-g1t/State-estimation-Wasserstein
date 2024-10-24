# SPDX-FileCopyrightText: The sinkhorn-rom authors, see file AUTHORS
#
# SPDX-License-Identifier: GPL-3.0-or-later

from geomloss import SamplesLoss
import torch
import numpy as np
import time
#from geomloss import ImagesBarycenter
from geomloss.sinkhorn_images import sinkhorn_divergence
from ..lib.DataManipulators.Problems import *
from ..lib.Evaluators.Barycenter import ImagesLoss, ImagesBarycenter_v2, projGraFixSupp
#from ..config import *
from ..config import results_dir, device, dtype
from ..visualization import plot_fields

import matplotlib.pyplot as plt
import matplotlib


# Load data from Gaussian1d Problem
problem_name = 'Burger2d'
problem = Problem(name = problem_name)
field_coordinates, snapshots, parameters, _ = problem.load_dataset(nparam=25)

params_sinkhorn_bary = {'blur': 0.001 , 'p':2, 'scaling_N':100,'backward_iterations':5} 
params_opt_best_barycenter = {'optimizer': 'LBFGS','eps_conv': 1.e-7,'nmax': 10}

# Initial example densities
a = snapshots[1].reshape(64,64)
b = snapshots[4].reshape(64,64)
c = snapshots[2].reshape(64,64)
d = snapshots[5].reshape(64,64)




firstSet  = torch.cat((a[None,None,:,:], c[None,None,:,:]), dim=0) 
secondSet = torch.cat((b[None,None,:,:], d[None,None,:,:]), dim=0) 

Distance_images= ImagesLoss(firstSet,secondSet,blur=0.001,scaling=0.8)
print('Distance with ImagesLoss',Distance_images)

# Let us compute Barycenter([a, b], weights=[0.3, 0.7])
weights_ref = torch.tensor([0.3, 0.7], dtype=dtype, device=device)
measures = torch.cat((a[None,None,:,:], b[None,None,:,:]), dim=1)  # fields are given per columns

print('measures',measures.shape)
# print(weights_ref.shape)

barycenters = ImagesBarycenter_v2(measures=measures, weights=weights_ref[None,:], blur=0.001, scaling_N = 300)

print('barycenter_ref type',type(barycenters))
print('barycenter_ref shape',barycenters.shape)

print('barycenter',barycenters.shape)
barycenters = barycenters.view(1, 1, a.shape[-1], a.shape[-1])


myvmax = max([a.max(),barycenters[0, 0].max(),b.max()])

# visualization
X = np.linspace(0, 10, a.shape[-1])
Y = np.linspace(0, 10, a.shape[-1])
fig, axs = plt.subplots(1, 3,figsize=(8.5, 5))

im1 = axs[0].pcolor(X,Y,a.detach().cpu().numpy(),vmin=0,vmax=myvmax,cmap='jet',shading='auto')
axs[0].set_title("a")
fig.colorbar(im1,ax=axs[0])

im2 =axs[1].pcolor(X,Y,barycenters[0, 0].detach().cpu().numpy(),vmin=0,vmax=myvmax,cmap='jet',shading='auto')
axs[1].set_title("barycenter_image")
fig.colorbar(im2,ax=axs[1])

im3 = axs[2].pcolor(X,Y,b.detach().cpu().numpy(),vmin=0,vmax=myvmax,cmap='jet',shading='auto')
axs[2].set_title("b")
fig.colorbar(im3, ax=axs[2])


fig.tight_layout()
plt.show()


# We take the previous barycenter function as the function for which we want
# to discover the best barycentric coordinates.
# We know that the best barycentric coordinates are weights_ref by construction.

# bestbary, weights, _ = projGraFixSupp(
#                         target_field=barycenters,
#                         measures = measures,
#                         field_coordinates=field_coordinates)

                        # params_opt = params_opt_best_barycenter,
                        # params_sinkhorn_bary=params_sinkhorn_bary,
#                         )
# # Print output weights
# print('Reference weights are {}. Computed weights are {}'.format(weights_ref, weights))


# fig, axs = plt.subplots(1, 2,figsize=(8.5, 5))

# im1 = axs[0].pcolor(X,Y,bestbary[0,0].detach().cpu().numpy(),vmin=0,vmax=myvmax,cmap='jet',shading='auto')
# axs[0].set_title("bestbary")
# fig.colorbar(im1,ax=axs[0])

# im2 =axs[1].pcolor(X,Y,barycenters[0, 0].detach().cpu().numpy(),vmin=0,vmax=myvmax,cmap='jet',shading='auto')
# axs[1].set_title("barycenter_image")
# fig.colorbar(im2,ax=axs[1])

# fig.tight_layout()
# plt.show()

