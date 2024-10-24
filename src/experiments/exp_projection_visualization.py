# SPDX-FileCopyrightText: The sinkhorn-rom authors, see file AUTHORS
#
# SPDX-License-Identifier: GPL-3.0-or-later

# import pykeops
# pykeops.clean_pykeops()
# print(pykeops.__version__)            # Should be 1.5
# pykeops.test_numpy_bindings()   # Should tell you sth like “bindings OK"
# pykeops.test_torch_bindings()     # Should tell you sth like “bindings OK"




#from geomloss import SamplesLoss
import torch
import numpy as np
import time
#from geomloss import ImagesBarycenter
#from geomloss.sinkhorn_images import sinkhorn_divergence
from ..lib.DataManipulators.Problems import *
from ..lib.Evaluators.Barycenter import  ImagesLoss, ImagesBarycenter_v2, projGraFixSupp, projGraAdapSupp, projRGraSP
from ..config import results_dir, device, dtype, use_pykeops
from ..visualization import plot_fields, plot_fields_images,plot_fields_images_v2
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

problem = Problem(name=args.p,id='Projection_out_id2_N100_GraAS_k5_nmax200_lr01',
                    config_set_fit={'nparam': args.nfit, 'id_set': args.idfit},
                    config_set_predict={'nparam': args.np, 'id_set': args.idp})




field_coordinates, snapshots, parameters, _ = problem.load_dataset(**problem.config_set_fit)
_, snapshots_test, _, _ = problem.load_dataset(**problem.config_set_predict)

root_dir = check_create_dir(results_dir+'{}/{}/'.format(problem.name, problem.id))

params_sinkhorn_bary = {'blur': 0.001 , 'p':2, 'scaling_N':100,'backward_iterations':5} 
params_opt_best_barycenter = {'optimizer': 'Adam','lr': 0.1,'nmax': 200,'gamma':1,'k_sparse':15}





mymeasures = torch.cat([snapshots[id][None,None,:,:] for id in range(args.nfit)], dim=1)

mytarget = torch.load(root_dir+'target',map_location=torch.device('cpu'))
best_weight = torch.load(root_dir+'best_weight',map_location=torch.device('cpu'))
best_bary = torch.load(root_dir+'best_bary',map_location=torch.device('cpu'))

# mytarget = snapshots_test[2][None,None,:,:]
# best_bary, best_weight, evolution = projRGraSP(
#                         target_field= mytarget,
#                         measures = mymeasures,
#                         field_coordinates=field_coordinates,
#                         params_opt = params_opt_best_barycenter,
#                         params_sinkhorn_bary=params_sinkhorn_bary,
#                         )


# # save best_bary and weights
# torch.save(best_weight, root_dir+'best_weight')
# torch.save(best_bary,root_dir+'best_bary')
# torch.save(mytarget,root_dir+'target')


# visualization best barycenter
my_fig_title = 'myprediction'
fig_opts={'colors': [], 'labels': ['ref', 'fit'], 'titles': None, 'plot_type': 'voronoi', 'rootdir': root_dir,  'fig_title': my_fig_title, 'format': '.pdf'}
plot_fields_images_v2(fields= [mytarget[0,0].detach().cpu().numpy(), best_bary[0,0].detach().cpu().numpy()],
                        spatial_coordinates= field_coordinates.cpu().numpy(),fig_opts=fig_opts,)


# Save weights
fig, ax= plt.subplots()
cp=ax.pcolormesh(best_weight.detach().cpu().numpy().reshape(-1,10),cmap='viridis',edgecolors='w', linewidths=2)
fig.colorbar(cp)
fig.tight_layout()
fig.savefig(root_dir+'best_weight2.pdf')


# # visualization evolution
# idims = 1 + np.arange(len(evolution['loss']))
# len_supp = [supp.shape[0] for supp in evolution['support']]
# print('total_loss',evolution['loss'])
# total_loss = evolution['loss']

# df_loss = pd.DataFrame(total_loss,columns =['loss'])
# df_supp = pd.DataFrame(len_supp,columns =['support'])
# df_loss.to_csv(root_dir+'loss'+'.csv')
# df_supp.to_csv(root_dir+'support'+'.csv')

# fig, axs = plt.subplots(1, 2,figsize=(8.5, 5))
# im1 = axs[0].plot(idims,total_loss,color='r', linestyle='-', marker='o', label='L')
# axs[0].set_yscale('log')
# axs[0].set_title("Loss")
# axs[0].set_xlabel('Iteration')


# im2 =axs[1].plot(idims,len_supp,color='b', linestyle='-', marker='o', label='S')
# axs[1].set_title("Support")
# axs[1].set_xlabel('Iteration')

# fig.tight_layout()
# fig.savefig(root_dir+ problem.name +'_evolution_gamma.pdf')
# #plt.show()

