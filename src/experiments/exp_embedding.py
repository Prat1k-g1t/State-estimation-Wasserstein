# SPDX-FileCopyrightText: The sinkhorn-rom authors, see file AUTHORS
#
# SPDX-License-Identifier: GPL-3.0-or-later

# import pykeops
# pykeops.clean_pykeops()
# print(pykeops.__version__)            # Should be 1.5
# pykeops.test_numpy_bindings()   # Should tell you sth like “bindings OK"
# pykeops.test_torch_bindings()     # Should tell you sth like “bindings OK"

from matplotlib.pyplot import legend
import seaborn as sns

#from geomloss import SamplesLoss
import argparse

from torch import Tensor

from ..lib.Models.predictionWeight import predictionWeight
from ..lib.Benchmarks.Benchmark import *
from ..lib.DataManipulators.Problems import *
from ..tools import get_dissimilarity_matrix, display_scaling_methods, fetch_dense, compute_scalings, find_knn, get_anisotropic_norm

from ..config import device

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
parser.add_argument('-nfit',type=int, default=100, help='number of parameters fit')
parser.add_argument('-idfit', type=int, default=0, help='id set fit')
parser.add_argument('-np', type=int, default=100, help='number of parameters predict')
parser.add_argument('-idp', type=int, default=1, help='id set predict')
args = parser.parse_args()

# Params for sinkhorn

#Loss = SamplesLoss("sinkhorn", blur=5.e-3, scaling=.9)
params_sinkhorn_bary = {'blur': 0.001, 'p':2, 'scaling_N':100,'backward_iterations':5} 
params_opt_best_barycenter = {'optimizer': 'Adam','lr': 0.01,'nmax': 100,'gamma':1,'k_sparse':10}
#params_KNN = {'n_neighbors':100}

#params_regression    = {'type_regression':'Shepard', 'reg_param': 5.e-3,'n_neighbors':20, 'adaptive_knn':False,'power':4}
#params_kernel       = {'metric':'rbf', 'gamma':10}

problem = Problem(name=args.p, id='LE_n100',
        config_set_fit={'nparam': args.nfit, 'id_set': args.idfit},
        config_set_predict={'nparam': args.np, 'id_set': args.idp})

root_dir = check_create_dir(results_dir+'{}/{}/'.format(problem.name, problem.id))

field_coordinates, snapshots, parameters, _ = problem.load_dataset(**problem.config_set_fit)
query_train = QueryDataStruct(parameter=parameters,field_spatial_coordinates=[field_coordinates])
target_train = TargetDataStruct(field=snapshots)

field_coordinates_test, snapshots_test, parameters_test, _ = problem.load_dataset(**problem.config_set_predict)
query_test = QueryDataStruct(parameter=parameters_test,field_spatial_coordinates=[field_coordinates_test])
target_test = TargetDataStruct(field=snapshots_test)
points = torch.Tensor(parameters.detach().cpu().numpy())
points_test = torch.Tensor(parameters_test.detach().cpu().numpy())



D, _ = get_dissimilarity_matrix(target_train)
sqdists = D.relu()

P = ((points[:, None, :] - points[None, :, :])**2).sum(-1)
fig =sns.jointplot(x=P.view(-1), y=D.view(-1))  
plt.subplots_adjust(left=0.1) 
plt.savefig(root_dir+'before.pdf')                                                                
plt.show()
U = compute_scalings(points, sqdists,  nits=1, k=90, robust=True, cvx=True)
#print(U)


dUd = get_anisotropic_norm(points,U)
fig =sns.jointplot(x=dUd.view(-1), y=D.view(-1))
fig.ax_joint.set_xlim(None,0.2)
fig.ax_joint.set_ylim(None,0.2)

plt.subplots_adjust(left=0.1)
plt.savefig(root_dir+'after.pdf')
plt.show()

# mypoint = points_test[4]
# mytarget = snapshots_test[4]

# print(mypoint)
# best_bary, best_weight, evolution = anisotropic_predict(points,snapshots,U,mypoint, mytarget,k=20,params_opt=params_opt_best_barycenter, params_sinkhorn_bary=params_sinkhorn_bary)



# # visualization best barycenter
# my_fig_title = 'prediction'
# fig_opts={'colors': [], 'labels': ['ref', 'fit'], 'titles': None, 'plot_type': 'voronoi', 'rootdir': root_dir,  'fig_title': my_fig_title, 'format': '.pdf'}
# plot_fields_images(fields= [mytarget.detach().cpu().numpy(), best_bary[0,0].detach().cpu().numpy()],
#                         spatial_coordinates= field_coordinates.cpu().numpy(),fig_opts=fig_opts,)


# # Save weights
# fig, ax= plt.subplots()
# cp=ax.pcolormesh(best_weight.detach().cpu().numpy().reshape(-1,10),cmap='jet',edgecolors='w', linewidths=2)
# fig.colorbar(cp)
# fig.tight_layout()
# fig.savefig(root_dir+'best_weight.pdf')


# # visualization evolution
# idims = 1 + np.arange(len(evolution['loss']))
# len_supp = [supp.shape[0] for supp in evolution['support']]
# print('total_loss',evolution['loss'])
# total_loss = evolution['loss']
# total_error = evolution['true_error']

# df_loss = pd.DataFrame(total_loss,columns =['loss'])
# df_supp = pd.DataFrame(len_supp,columns =['support'])
# df_error = pd.DataFrame(total_error,columns =['error'])
# df_loss.to_csv(root_dir+'loss'+'.csv')
# df_supp.to_csv(root_dir+'support'+'.csv')
# df_error.to_csv(root_dir+'error'+'.csv')

# fig, axs = plt.subplots(1, 3,figsize=(8.5, 8))
# im1 = axs[0].plot(idims,total_loss,color='r', linestyle='-', marker='o', label='L')
# axs[0].set_yscale('log')
# axs[0].set_title("Loss")
# axs[0].set_xlabel('Iteration')


# im2 =axs[1].plot(idims,len_supp,color='b', linestyle='-', marker='o', label='S')
# axs[1].set_title("Support")
# axs[1].set_xlabel('Iteration')

# im3 = axs[2].plot(idims,total_error,color='g', linestyle='-', marker='o', label='E')
# axs[2].set_yscale('log')
# axs[2].set_title("Error")
# axs[2].set_xlabel('Iteration')

# fig.tight_layout()
# fig.savefig(root_dir+ problem.name +'_evolution_gamma.pdf')
# #plt.show()

















#=================================================================================================================
#print('D_ij',D_ij)

# indices = find_knn(D_ij,k=5)
# print('indices',indices)

# indices = find_knn_ball(D_ij,eps=0.1)
# print('indices',indices)
###=====================================
# points = torch.Tensor([
#     [0., 1.],  # A
#     [1., 1.],  # B
#     [0., 0.],  # C
#     [1., 0.],  # D
# ])

# sqdists = torch.Tensor([
#     # A, B,  C,  D
#     [0., 1., 1., 1.],  # A
#     [1., 0., 1., 1.],  # B
#     [1., 1., 0., 1.],  # C
#     [1., 1., 1., 0.],  # D
# ])


# from sklearn.manifold import MDS

# fields_embedding = MDS(n_components=2, dissimilarity="precomputed")
# params_embedding = MDS(n_components=2)

# x_train = fields_embedding.fit_transform(dissimilarity_matrix.relu().sqrt().cpu().numpy())
# p_train = params_embedding.fit_transform(parameters.cpu().numpy()) 
# points = torch.Tensor(p_train)



#display_scaling_methods(compute_scalings, points,fetch_dense(sqdists), k=4, scale=.05, axes = [-.2, .2, -.2, .2])
#display_scaling_methods(compute_scalings, points,sqdists, k=4, scale=.05, axes = [-.2, .2, -.2, .2])



# U = compute_scalings(points, sqdists, k=20, nits=1, robust=True, cvx=True)
# print(U)

# P_ij = ((points[:, None, :] - points[None, :, :])**2).sum(-1)
# sns.jointplot(x=P_ij.sqrt().view(-1), y=D_ij.sqrt().view(-1))
# plt.show()

# dUd_ij = get_anisotropic_norm(points,U)
# print(dUd_ij)
# #dUd = anisotropic_norm(points,U)

# sns.jointplot(x=dUd_ij.sqrt().view(-1), y=D_ij.sqrt().view(-1))
# plt.show()

# dU = get_affine_invariant_distance_matrix(U)
# sns.jointplot(x=dUd_ij.sqrt().view(-1), y=dU.sqrt().view(-1))
# plt.show()



#D_ij_test= get_distance_matrix(snapshots,snapshots_test)

# dUd_ij_test = get_anisotropic_norm_v2(points,points_test,U)
# sns.jointplot(x=dUd_ij_test.sqrt().view(-1), y=D_ij_test.sqrt().view(-1))
# plt.show()








#===================================================================================================================


# P = ((points[:, None, :] - points[None, :, :])**2).sum(-1)
# fig =sns.jointplot(x=P.sqrt().view(-1), y=D.sqrt().view(-1))
# fig.savefig(root_dir+ problem.name +'_W_P.pdf')
# plt.show()
# # sort indices
# indices_sort = torch.argsort(D,dim=1)
# #print(indices_sort)
# U = compute_scalings(points, sqdists, k=190, nits=1, robust=True, cvx=True)
# dUd = get_anisotropic_norm(points,U)
# dU = get_affine_invariant_distance_matrix(U)

# fig =sns.jointplot(x=dUd.sqrt().view(-1), y=D.sqrt().view(-1))
# fig.savefig(root_dir+ problem.name +'_W_dUd_k' +'.pdf')
# fig = sns.jointplot(x=dUd.sqrt().view(-1), y=dU.sqrt().view(-1))
# fig.savefig(root_dir+ problem.name +'_dU_dUd_k' +'.pdf')


# knn_params = [10,20,40,80,160]
# list_mean_distance =[]
# list_max_distance =[]
# for k in knn_params:
#     #U = compute_scalings(points, sqdists, k=k, nits=1, robust=True, cvx=True)
#     #dUd = get_anisotropic_norm(points,U)
#     #dU = get_affine_invariant_distance_matrix(U)
#     indices = indices_sort[:,1:k+1]
#     #print('indices',indices)
#     list_RdU = []
#     for i in range(dU.shape[0]):
#         dU_i = dU[i][indices[i]]
#         list_RdU.append(dU_i)
#     RdU = torch.vstack(list_RdU)
#     mean_distance = torch.mean(RdU,dim=1)
#     max_distance  = torch.max(RdU,dim=1).values
#     list_mean_distance.append(mean_distance.numpy())
#     list_max_distance.append(max_distance.numpy())



#     # Plot figures
#     # fig =sns.jointplot(x=dUd.sqrt().view(-1), y=D.sqrt().view(-1))
#     # fig.savefig(root_dir+ problem.name +'_W_dUd_k' +str(k)+'.pdf')
#     # fig = sns.jointplot(x=dUd.sqrt().view(-1), y=dU.sqrt().view(-1))
#     # fig.savefig(root_dir+ problem.name +'_dU_dUd_k' +str(k)+'.pdf')


# print(np.array(list_mean_distance))

# mat_mean_errors = np.transpose(np.array(list_mean_distance))

# print('errors',mat_mean_errors.shape)
# legends=['k=10','k=20','k=40','k=80','k=160']
# df_mean = pd.DataFrame(data=mat_mean_errors, columns=legends)
# plt.figure(figsize=(10, 5))
# #sns.boxplot(data=df_mean)
# sns.set(style="whitegrid")
# sns.swarmplot(data=df_mean,size=5,marker="D",alpha=.8,edgecolor="gray")
# #sns.stripplot(data=df)
# #sns.pointplot(data=df)
# #sns.violinplot(data=df)
# #grid = sns.pairplot(data=df)
# # plt.xscale('log')
# plt.yscale('log')
# plt.savefig(root_dir+ problem.name +'Mean' +'.pdf')
# plt.show()


# mat_max_errors = np.transpose(np.array(list_max_distance))
# df_max = pd.DataFrame(data=mat_max_errors, columns=legends)
# plt.figure(figsize=(10, 5))
# #sns.boxplot(data=df_max)
# sns.set(style="whitegrid")
# sns.swarmplot(data=df_max,size=5,marker="D",alpha=.8,edgecolor="gray")
# #sns.stripplot(data=df)
# #sns.pointplot(data=df)
# #sns.violinplot(data=df)
# #grid = sns.pairplot(data=df)
# # plt.xscale('log')
# plt.yscale('log')
# plt.savefig(root_dir+ problem.name +'Max' +'.pdf')
# plt.show()