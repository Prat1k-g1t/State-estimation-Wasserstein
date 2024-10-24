# SPDX-FileCopyrightText: The sinkhorn-rom authors, see file AUTHORS
#
# SPDX-License-Identifier: GPL-3.0-or-later

# import pykeops
# pykeops.clean_pykeops()
# print(pykeops.__version__)            # Should be 1.5
# pykeops.test_numpy_bindings()   # Should tell you sth like “bindings OK"
# pykeops.test_torch_bindings()     # Should tell you sth like “bindings OK"

from matplotlib.pyplot import legend
from matplotlib import ticker, cm
import seaborn as sns

#from geomloss import SamplesLoss
import argparse
#from sympy import N

import torch

from ..lib.Benchmarks.Benchmark import *
from ..lib.DataManipulators.Problems import *
from ..lib.Evaluators.Barycenter import  ImagesLoss, ImagesBarycenter_v2, projGraFixSupp, projGraAdapSupp, projRGraSP, mat2simplex
from ..tools import find_KNN_snapshots, barycentric_coordinates, best_weight_Dirac_masses
from ..config import device, dtype
from ..visualization import plot_fields_images, plot_evolution, plot_Loss_map
#from matplotlib.patches import Polygon
#from shapely.geometry import Point, Polygon
import numpy as np
from tqdm import tqdm

print('check cuda', torch.cuda.is_available())


# List of available problems
problem_dict = {'Gaussian1d': Gaussian1d,
                'Gaussian2d': Gaussian2d,
                'VlasovPoisson': VlasovPoisson,
                'Transport1d': Transport1d,
                'Burger2d'   : Burger2d
                }

# Parse
parser = argparse.ArgumentParser()
parser.add_argument('-p', type=str, default='Burger2d', help=', '.join(problem_dict.keys()))
parser.add_argument('-nfit', type=int, default=200, help='number of parameters fit')
parser.add_argument('-idfit', type=int, default=0, help='id set fit')
parser.add_argument('-np', type=int, default=100, help='number of parameters predict')
parser.add_argument('-idp', type=int, default=1, help='id set predict')
args = parser.parse_args()

# Params for sinkhorn

#Loss = SamplesLoss("sinkhorn", blur=5.e-3, scaling=.9)
params_sinkhorn_bary = {'blur': 0.001, 'p':2, 'scaling_N':100,'backward_iterations':5} 
params_opt_best_barycenter = {'optimizer': 'Adam','lr': 0.02,'nmax': 300, 'type_loss':'W2','gamma':1,'k_sparse':10}

#params_KNN = {'n_neighbors':100}
#params_regression    = {'type_regression':'Shepard', 'reg_param': 5.e-3,'n_neighbors':20, 'adaptive_knn':False,'power':4}
#params_kernel       = {'metric':'rbf', 'gamma':10}

problem = Problem(name=args.p, id='Lossmap_train200_bary3_W2_lr002_nmax300',
        config_set_fit={'nparam': args.nfit, 'id_set': args.idfit},
        config_set_predict={'nparam': args.np, 'id_set': args.idp})

root_dir = check_create_dir(results_dir+'{}/{}/'.format(problem.name, problem.id))

field_coordinates_train, snapshots_train, parameters_train, _ = problem.load_dataset(**problem.config_set_fit)
query_train = QueryDataStruct(parameter=parameters_train,field_spatial_coordinates=[field_coordinates_train])
target_train = TargetDataStruct(field=snapshots_train)

field_coordinates_test, snapshots_test, parameters_test, _ = problem.load_dataset(**problem.config_set_predict)
query_test = QueryDataStruct(parameter=parameters_test,field_spatial_coordinates=[field_coordinates_test])
target_test = TargetDataStruct(field=snapshots_test)

points_train = torch.Tensor(parameters_train.detach().cpu().numpy())
points_test = torch.Tensor(parameters_test.detach().cpu().numpy())




# Test 1
# X = [np.array([0., 0]), np.array([1, 1]), np.array([0, 1])]
# polygon = Polygon(X)
# delta = 0.005
# xv = yv = np.arange(0., 1., delta)
# Xgrid, Ygrid = np.meshgrid(xv, yv)
# Z = np.zeros(Xgrid.shape)

# x = torch.tensor(np.array([0.3, 0.7]), dtype=dtype, device=device)
# interior_polygon = np.zeros(Xgrid.shape)
# for index, _ in np.ndenumerate(Xgrid):
#     interior_polygon[index] = polygon.contains(Point(Xgrid[index], Ygrid[index]))
#     if interior_polygon[index]==1:
#         point = np.array([Xgrid[index], Ygrid[index]])
#         W= barycentric_coordinates(X, point)

#         weight = torch.tensor(W,dtype=dtype,device=device)
#         bary = torch.matmul(weight,main_points)

#         Z[index] = 3*mse(torch.norm(x-main_points,dim=1).pow(2), torch.norm(bary-main_points,dim=1).pow(2)).item()
# Z[interior_polygon == 0] = np.ma.masked

# # test 2: optimal weight with PGD
# best_bary,best_weight,evolution = best_weight(main_points,x,params_opt_best_barycenter)
# evolution_weights = torch.vstack(evolution['weight'])
# print(evolution_weights)
# evolution_points = torch.matmul(evolution_weights,main_points)
# evolution_losses = torch.tensor(evolution['loss'])
# print(evolution_losses)

# mypoints= evolution_points.cpu().numpy()
# myloss = evolution_losses.cpu().numpy()
# Loss = np.true_divide(myloss, np.max(myloss))
# # create map color
# indexColor = np.floor(Loss*255)
# cmap = plt.cm.get_cmap('jet')
# colors = cmap(np.arange(cmap.N))

# fig, ax = plt.subplots()
# nlevels = 5
# levels = np.logspace(-3, 1, num=nlevels)
# cs = ax.contourf(Xgrid, Ygrid, Z,levels, locator=ticker.LogLocator(), cmap='jet') # locator=ticker.LogLocator()
# ax.plot(*x.cpu().numpy(), marker="o", markersize=10, markeredgecolor="red", markerfacecolor="red")
# cbar = fig.colorbar(cs)
# for i in range(mypoints.shape[0]):
#     ax.plot(mypoints[i][0],mypoints[i][1],'o',color=colors[int(indexColor[i])])
# plt.show()


# Test 2

# Barycenter with support =2


n_size=3
main_points = torch.tensor(np.array([[0.,0.],[1.,1.],[0.,1.]]),dtype=dtype,device=device)
main_weights = torch.tensor(np.eye(n_size),dtype=dtype,device=device)

measures = torch.cat([snapshots_train[id][None,None,:,:] for id in range(n_size)], dim=1)
total_measures = torch.cat([snapshots_train[id][None,None,:,:] for id in range(n_size)], dim=0)

weights_ref = torch.tensor([0.3, 0.1, 0.6], dtype=dtype, device=device)
target = ImagesBarycenter_v2(measures=measures, weights=weights_ref[None,:], blur=0.001, scaling_N = 300,backward_iterations=5)
targets =  target.repeat(n_size, 1, 1, 1)
distance_true = ImagesLoss(targets,total_measures,blur=0.001,scaling=0.8)
# epsilon = 0.001
# noise = epsilon*torch.rand(n_size,dtype=dtype,device=device)
# distance_noise = distance_true + noise

point_ref = torch.matmul(weights_ref[None,:],main_points)
fig_opts = {'rootdir':root_dir,'format':'.pdf'}




# On test set 
# myindex = 1
# target = snapshots_test[myindex][None,None,:,:]
# distance,index_neighbors = find_KNN_snapshots(snapshots_train, snapshots_test[myindex],k=n_size)
# indices = index_neighbors.flatten()
# distance = distance.to(dtype=dtype)

# targets =  target.repeat(n_size, 1, 1, 1)
# measures = torch.cat([snapshots_train[indices[id]][None,None,:,:] for id in range(n_size)], dim=1)
# total_measures = torch.cat([snapshots_train[indices[id]][None,None,:,:] for id in range(n_size)], dim=0)
# distance_true = ImagesLoss(targets,total_measures,blur=0.001,scaling=0.8)


# best_ref,weights_ref,evolution_ref = projGraFixSupp(target_field= mytarget,measures = mymeasures,field_coordinates=field_coordinates_train,
#                         params_opt = {'optimizer': 'Adam','lr': 0.2,'nmax': 200,'gamma':1,'k_sparse':10},
#                         params_sinkhorn_bary=params_sinkhorn_bary)
# print('loss_ref',evolution_ref['loss'])
# point_ref = torch.matmul(weights_ref[None,:],main_points)
# print('point_ref',point_ref)







##=========================================================
#X = [np.array([0., 0]), np.array([1, 1]), np.array([0, 1])]
# polygon = Polygon(X)
# xv = yv = np.linspace(0., 1., 20)
# Xgrid, Ygrid = np.meshgrid(xv, yv)

# Z = np.zeros(Xgrid.shape)

# interior_polygon = np.zeros(Xgrid.shape)
# for index, _ in np.ndenumerate(Xgrid):
#     interior_polygon[index] = polygon.contains(Point(Xgrid[index], Ygrid[index]))
#     if interior_polygon[index]==1:
#         point = np.array([Xgrid[index], Ygrid[index]])
#         W= barycentric_coordinates(X, point)
#         weight = torch.tensor(W,dtype=dtype,device=device)
#         bary = ImagesBarycenter_v2(measures=measures, weights=weight[None,:], blur=0.001, scaling_N = 300,backward_iterations=5)

#         barycenters = bary.repeat(n_size,1,1,1)
#         distance_approx = ImagesLoss(barycenters,total_measures,blur=0.001,scaling=0.8)
#         loss = n_size*mse(distance_true, distance_approx)

#         Z[index] = loss.item()
# Z[interior_polygon == 0] = np.ma.masked

# fig, ax = plt.subplots()
# nlevels = 5
# levels = np.logspace(-5, -1, num=nlevels)
# cs = ax.contourf(Xgrid, Ygrid, Z,levels, locator=ticker.LogLocator(), cmap='jet') # locator=ticker.LogLocator()
# ax.plot(*point_ref[0].cpu().numpy(), marker="o", markersize=10, markeredgecolor="red", markerfacecolor="red")
# cbar = fig.colorbar(cs)
# plt.show()


## ===============================================================================================================================

best_bary, best_weight, evolution = projGraFixSupp(target_field= target,measures = measures,
                                    field_coordinates=field_coordinates_train,
                                    params_opt = params_opt_best_barycenter,
                                    params_sinkhorn_bary=params_sinkhorn_bary,distance_target_measures=distance_true)




plot_Loss_map(main_points,main_weights,target,measures,type_loss=params_opt_best_barycenter['type_loss'],distance_target_measures=distance_true, 
            point_ref=point_ref,evolution=evolution,fig_opts=fig_opts)



# ###=====================================================================
# save best_bary and weights
torch.save(best_weight, root_dir+'best_weight')
torch.save(best_bary,root_dir+'best_bary')
torch.save(target,root_dir+'target')


# visualization best barycenter
my_fig_title = 'prediction'
fig_opts={'colors': [], 'labels': ['ref', 'fit'], 'titles': None, 'plot_type': 'voronoi', 'rootdir': root_dir,  'fig_title': my_fig_title, 'format': '.pdf'}
plot_fields_images(fields= [target[0,0].detach().cpu().numpy(), best_bary[0,0].detach().cpu().numpy()],
                    spatial_coordinates= field_coordinates_train.cpu().numpy(),fig_opts=fig_opts,)


# Save weights
fig, ax= plt.subplots()
cp=ax.pcolormesh(best_weight.detach().cpu().numpy().reshape(-1,n_size),cmap='jet',edgecolors='w', linewidths=2)
fig.colorbar(cp)
fig.tight_layout()
fig.savefig(root_dir+'best_weight.pdf')

plot_evolution(evolution,fig_opts={'rootdir':root_dir,'format':'.pdf'})



















