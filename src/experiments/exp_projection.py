# SPDX-FileCopyrightText: The sinkhorn-rom authors, see file AUTHORS
#
# SPDX-License-Identifier: GPL-3.0-or-later

import pykeops
pykeops.clean_pykeops()
print(pykeops.__version__)            # Should be 1.5
pykeops.test_numpy_bindings()   # Should tell you sth like “bindings OK"
pykeops.test_torch_bindings()     # Should tell you sth like “bindings OK"




#from geomloss import SamplesLoss
import torch
import numpy as np
import time
#from geomloss import ImagesBarycenter
#from geomloss.sinkhorn_images import sinkhorn_divergence
from ..lib.DataManipulators.Problems import *
from ..lib.Evaluators.Barycenter import  ImagesLoss, ImagesBarycenter_v2, projGraFixSupp, projGraAdapSupp, projRGraSP
from ..config import results_dir, device, dtype, use_pykeops
from ..visualization import plot_fields, plot_fields_images
from ..utils import check_create_dir
import pandas as pd
import argparse
import matplotlib.pyplot as plt

# List of available problems
problem_dict = {'Gaussian1d': Gaussian1d,
                'Gaussian2d': Gaussian2d,
                'VlasovPoisson': VlasovPoisson,
                'Shallow_Water2d': Shallow_Water2d,
                }

parser = argparse.ArgumentParser()
parser.add_argument('-p', type=str, default='Burger2d', help=', '.join(problem_dict.keys()))
parser.add_argument('-nfit', type=int, default=100, help='number of parameters fit')
parser.add_argument('-idfit', type=int, default=0, help='id set fit')
parser.add_argument('-np', type=int, default=100, help='number of parameters predict')
parser.add_argument('-idp', type=int, default=1, help='id set predict')
args = parser.parse_args()


# Load data 

problem = Problem(name=args.p,id='Projection_out_id2_N100_GraSP_k15_nmax200_lr01',
                    config_set_fit={'nparam': args.nfit, 'id_set': args.idfit},
                    config_set_predict={'nparam': args.np, 'id_set': args.idp})




field_coordinates, snapshots, parameters, _ = problem.load_dataset(**problem.config_set_fit)
_, snapshots_test, _, _ = problem.load_dataset(**problem.config_set_predict)

root_dir = check_create_dir(results_dir+'{}/{}/'.format(problem.name, problem.id))

params_sinkhorn_bary = {'blur': 0.001 , 'p':2, 'scaling_N':100,'backward_iterations':5} 
params_opt_best_barycenter = {'optimizer': 'Adam','lr': 0.1,'nmax': 200,'gamma':1,'k_sparse':15}


# print('X',field_coordinates)
# print('params',parameters)
# print('fields',snapshots[0].shape)

# Params for sinkhorn
# eps = 5.e-3
# Loss = SamplesLoss("sinkhorn", blur=0.001, scaling=.8)



# # Barycenter with support =2
# # Initial example densities
# a = snapshots[0]
# b = snapshots[2]

# # Let us compute Barycenter([a, b], weights=[0.3, 0.7])
# weights_ref = torch.tensor([0.3, 0.7], dtype=dtype, device=device)
# measures = torch.cat((a[None,None,:,:], b[None,None,:,:]), dim=1)  # fields are given per columns


# # Barycenter with support =10

# #measures = torch.cat([snapshots[id][None,None,:,:] for id in np.arange(1,20,2).tolist()], dim=1)

# measures = torch.cat([snapshots[id][None,None,:,:] for id in range(10)], dim=1)
# weights_ref = torch.tensor([0.05, 0.1, 0.05,0.2,0.05,0.05,0.02,0.08,0.1,0.3], dtype=dtype, device=device)



# # # print('measures',measures.shape)
# # # print(weights_ref.shape,weights_ref.sum())

# barycenters = ImagesBarycenter_v2(measures=measures, weights=weights_ref[None,:], blur=0.001, scaling_N = 300,backward_iterations=5)
# #barycenters = barycenters.view(1, 1, a.shape[-2], a.shape[-1])

# # print('a',a.detach().cpu().numpy().sum())
# # print('b',b.detach().cpu().numpy().sum())
# print('bary_shape',barycenters.shape)
# print('bary', barycenters[0, 0].detach().cpu().numpy().sum())




# Distance_images_3 = ImagesLoss(b[None,None,:,:],barycenters,blur=0.001,scaling=0.9)
# print('Distance with ImagesLoss between b and bary',Distance_images_3)

# bary center :
# myparams_sinkhorn_bary = {'logsumexp': logsumexp_template(field_coordinates.shape[1], field_coordinates.shape[1]) if use_pykeops else None,
#                         'spatial_metric': 2,
#                         'eps': eps,
#                         'tau1': 100.,
#                         'tau2': 100.,
#                         'nmax': 100}

# bary_ref = barycenter(fields = [a.flatten(), b.flatten()],
#                     weights=weights_ref,
#                     field_coordinates=field_coordinates, **myparams_sinkhorn_bary)

# print('bary_ref',bary_ref.detach().cpu().numpy().sum())




# myvmax = max([a.max(),barycenters[0, 0].max(),b.max()]).cpu()

# # visualization
# X = np.linspace(0, 1, a.shape[-1])
# Y = np.linspace(0, 1, a.shape[-1])

# fig, axs = plt.subplots(1, 3,figsize=(8.5, 5))

# im1 = axs[0].contourf(X,Y,a.detach().cpu().numpy(),vmin=0,vmax=myvmax,cmap='jet')
# axs[0].set_title("a")
# fig.colorbar(im1,ax=axs[0])

# im2 =axs[1].contourf(X,Y,barycenters[0, 0].detach().cpu().numpy(),vmin=0,vmax=myvmax,cmap='jet')
# axs[1].set_title("barycenter_image")
# fig.colorbar(im2,ax=axs[1])
# # im3 = axs[0, 1].pcolor(X,Y,bary_ref.reshape(64,64).detach().cpu().numpy().T,vmin=0,vmax=myvmax,cmap='jet',shading='auto')
# #axs[0, 1].set_title("barycenter_Sinkhorn")
# #fig.colorbar(im3, ax=axs[0, 1])
# im3 = axs[2].contourf(X,Y,b.detach().cpu().numpy(),vmin=0,vmax=myvmax,cmap='jet')
# axs[2].set_title("b")
# fig.colorbar(im3, ax=axs[2])

# #fig.subplots_adjust(bottom=0.1, top=0.9, left=0.1, right=0.8,wspace=0.02, hspace=0.02)
# #cb_ax = fig.add_axes([0.83, 0.1, 0.02, 0.8])
# #cbar = fig.colorbar(im, ax=axs.ravel().tolist(), shrink=0.95)
# fig.tight_layout()
# fig.savefig(root_dir+ problem.name +'_bary_ref.pdf')
# plt.show()





# We take the previous barycenter function as the function for which we want
# to discover the best barycentric coordinates.
# We know that the best barycentric coordinates are weights_ref by construction.



mymeasures = torch.cat([snapshots[id][None,None,:,:] for id in range(args.nfit)], dim=1)
mytarget = snapshots_test[2][None,None,:,:]
#mytarget = barycenters
best_bary, best_weight, evolution = projRGraSP(
                        target_field= mytarget,
                        measures = mymeasures,
                        field_coordinates=field_coordinates,
                        params_opt = params_opt_best_barycenter,
                        params_sinkhorn_bary=params_sinkhorn_bary,
                        )
# Print output weights
#print('Reference weights are {}. Computed weights are {}'.format(weights_ref, best_weight))


# save best_bary and weights
torch.save(best_weight, root_dir+'best_weight')
torch.save(best_bary,root_dir+'best_bary')
torch.save(mytarget,root_dir+'target')


# visualization best barycenter
my_fig_title = 'prediction'
fig_opts={'colors': [], 'labels': ['ref', 'fit'], 'titles': None, 'plot_type': 'voronoi', 'rootdir': root_dir,  'fig_title': my_fig_title, 'format': '.pdf'}
plot_fields_images(fields= [mytarget[0,0].detach().cpu().numpy(), best_bary[0,0].detach().cpu().numpy()],
                        spatial_coordinates= field_coordinates.cpu().numpy(),fig_opts=fig_opts,)


# Save weights
fig, ax= plt.subplots()
cp=ax.pcolormesh(best_weight.detach().cpu().numpy().reshape(-1,10),cmap='jet',edgecolors='w', linewidths=2)
fig.colorbar(cp)
fig.tight_layout()
fig.savefig(root_dir+'best_weight.pdf')


# visualization evolution
idims = 1 + np.arange(len(evolution['loss']))
len_supp = [supp.shape[0] for supp in evolution['support']]
print('total_loss',evolution['loss'])
total_loss = evolution['loss']

df_loss = pd.DataFrame(total_loss,columns =['loss'])
df_supp = pd.DataFrame(len_supp,columns =['support'])
df_loss.to_csv(root_dir+'loss'+'.csv')
df_supp.to_csv(root_dir+'support'+'.csv')

fig, axs = plt.subplots(1, 2,figsize=(8.5, 5))
im1 = axs[0].plot(idims,total_loss,color='r', linestyle='-', marker='o', label='L')
axs[0].set_yscale('log')
axs[0].set_title("Loss")
axs[0].set_xlabel('Iteration')


im2 =axs[1].plot(idims,len_supp,color='b', linestyle='-', marker='o', label='S')
axs[1].set_title("Support")
axs[1].set_xlabel('Iteration')

fig.tight_layout()
fig.savefig(root_dir+ problem.name +'_evolution_gamma.pdf')
#plt.show()

