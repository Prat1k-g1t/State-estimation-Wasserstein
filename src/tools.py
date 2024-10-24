# SPDX-FileCopyrightText: The sinkhorn-rom authors, see file AUTHORS
#
# SPDX-License-Identifier: GPL-3.0-or-later

from socket import NI_MAXSERV
import torch
from torch.utils.data import Dataset, DataLoader
import itertools
from collections import Counter
from sklearn import preprocessing
from sklearn.neighbors import NearestNeighbors
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer
from .config import dtype, device
import pykeops
from pykeops.torch import LazyTensor
from tqdm import tqdm
from .lib.DataManipulators.DataStruct import QueryDataStruct,TargetDataStruct

from .lib.Evaluators.Barycenter import ImagesLoss, ImagesBarycenter_v2, mat2simplex

#=====================================================================
def load_data(x, folder="prepared_data/data/", weights=True):
    return {
        # Parameters for the solver:
        "params": torch.load(f"{folder}{x}/params"),
        # Solution, supported on a grid:
        "fields": torch.load(f"{folder}{x}/fields"),
        # Coordinates of the grid points:
        "points": torch.load(f"{folder}{x}/points"),
        # Barycentric weights:
        "weights": torch.load(f"prepared_data/weights/{x}/weights")
        if weights
        else None,
    }


def display_data(params, fields, labels=True):
    plt.figure(figsize=(8, 8))
    for i in range(5):
        for j in range(5):
            n = i + j * 5
            plt.subplot(5, 5, n + 1)
            plt.imshow(fields[n])
            plt.axis("off")
            if labels:
                plt.title(
                    f"{n}:\n" + ",".join("{:.1f}".format(10 * x) for x in params[n])
                )


def fetch_dense(sqdists_ij):
    def fetch(nn_indices):
        """Lookup the matrix sqdists_ij.

        nn_indices is an (N, K) array of integers,

        We return an (N, K) array of values.
        """
        # Input shape
        N, k = nn_indices.shape
        # nn_indices[i, j] is the index of the j-th neighbor of i.
        offsets = N * torch.arange(N).type_as(nn_indices)
        indices = nn_indices + offsets.view(N, 1)  # (N, K)

        # Fetch the relevant Wasserstein distances:
        nn_sqwass = sqdists_ij.view(-1)[indices.view(-1)]
        return nn_sqwass.view(N, k)

    return fetch


from matplotlib.collections import LineCollection


def scaling_ellipses(points, Scalings_emb, scale=1):
    Semb_eig, Semb_axes = torch.linalg.eigh(Scalings_emb)  # (N, 2), (N, 2, 2)
    Semb_scales = 1 / (Semb_eig.relu().sqrt() + 1e-4)  # (N, 2)
    # print(Semb_eig)

    # We sample our ellipse glyphs with 32 points:
    t = torch.linspace(0, 2 * np.pi, 33).type_as(Semb_axes)  # (R=32+1)
    circle = torch.stack((t.cos(), t.sin())).T.view(-1, 2)  # (R, 2)

    # (N,2,2) "@" (R,2) "@" (N,2) -> (N,R,2)
    Semb_ell = torch.einsum("ndk,rk,nk->nrd", Semb_axes, circle, Semb_scales)
    Semb_ell = points.view(-1, 1, 2) + scale * Semb_ell

    return Semb_ell


def display_embedding(coords, ellipses=None):
    plt.scatter(coords[:, 0], coords[:, 1], c="red", marker="x")

    if ellipses is not None:
        line_segments = LineCollection(
            ellipses, linewidths=0.5, colors="blue", linestyle="solid"
        )
        plt.gca().add_collection(line_segments)

    for i, (x, y) in enumerate(coords):
        plt.text(
            x,
            y,
            f"{i}",
            fontsize=15,
            horizontalalignment="center",
            verticalalignment="center",
        )
    plt.axis("equal")


def display_scaling_its(
    compute_scalings, points, fetch_Dij, axes=None, scale=1, **kwargs
):
    plt.figure(figsize=(12, 8))

    for i in range(6):
        Scalings_emb = compute_scalings(points, fetch_Dij, nits=i, **kwargs)
        ellipses = scaling_ellipses(points, Scalings_emb, scale=scale)

        plt.subplot(2, 3, i + 1)
        display_embedding(points, ellipses=ellipses)
        if axes is not None:
            plt.axis(axes)
        plt.title(f"Local scalings, nits={i}")


def display_scaling_methods(
    compute_scalings, points, fetch_Dij, k=10, scale=1, axes=None
):
    plt.figure(figsize=(8, 8))
    for (i, cvx) in enumerate([False, True]):
        for (j, robust) in enumerate([False, True]):

            Scalings_emb = compute_scalings(points, fetch_Dij, k=k, nits=5, robust=robust, cvx=cvx)
            ellipses = scaling_ellipses(points, Scalings_emb, scale=scale)

            plt.subplot(2, 2, 2 * i + j + 1)
            display_embedding(points, ellipses=ellipses)
            if axes is not None:
                plt.axis(axes)
            plt.title(f"cvx={cvx}, robust={robust}")
    plt.show()






#====================================================================

def get_dissimilarity_matrix(target: TargetDataStruct):
    class PairWiseDataset(Dataset):
        def __init__(self, target: TargetDataStruct):
            self.images_source=[]
            self.images_target=[]
            for ((i,ti), (j,tj)) in itertools.combinations(enumerate(target), 2):
                self.images_source.append((i,ti.field[0][None,:,:]))
                self.images_target.append((j,tj.field[0][None,:,:]))

        def __len__(self):
            return len(self.images_source)
        def __getitem__(self,idx):
            return self.images_source[idx], self.images_target[idx]

    myDataset = PairWiseDataset(target)

    Total_loss = []
    firstIndex =[]
    secondIndex=[]
    train_dataloader = DataLoader(myDataset, batch_size=300, shuffle=False)
    #train_features, train_labels = next(iter(train_dataloader))
    for i_batch, sample_batched in enumerate(train_dataloader):
        train_features, train_labels = sample_batched
        loss = ImagesLoss(train_features[1], train_labels[1],blur=0.0001,scaling=0.9)
        Total_loss.append(loss)
        firstIndex.append(train_features[0])
        secondIndex.append(train_labels[0])
    wasserstein_distances = torch.hstack(Total_loss)
    wasserstein_firstIndex   = torch.hstack(firstIndex)
    wasserstein_secondIndex = torch.hstack(secondIndex)
    #pair_indexes = [(u,v) for (u,v) in zip(wasserstein_firstIndex.tolist(),wasserstein_secondIndex.tolist())]
    pairwise_distance_tool = {'distance':wasserstein_distances, 'firstIndex':wasserstein_firstIndex, 'secondIndex':wasserstein_secondIndex}
    n_snapshots = len(target)
    dissimilarity_matrix = torch.zeros(n_snapshots, n_snapshots)
    
    for  id in range(wasserstein_distances.shape[0]):
        dissimilarity_matrix[wasserstein_firstIndex[id]][wasserstein_secondIndex[id]] = wasserstein_distances[id]
        dissimilarity_matrix[wasserstein_secondIndex[id]][wasserstein_firstIndex[id]] = wasserstein_distances[id]
    return dissimilarity_matrix, pairwise_distance_tool

def fetch_sqwass(distance_matrix,nn_indices):
    """Lookup the matrix sqdists_ij.

        nn_indices is an (N, K) array of integers,

        We return an (N, K) array of values.
        """
    N, K = nn_indices.shape
    R = torch.zeros(N,K)
    for i in range(N):
        for j in range(K):
            R[i][j]= distance_matrix[i][nn_indices[i,j]]
    return R




def restriction_distance_matrix(distance_matrix,index):
    R = torch.zeros(index.shape[0],index.shape[0])
    for i in range(index.shape[0]):
        for j in range(index.shape[0]):
            R[i][j]= distance_matrix[index[i]][index[j]]
    return R


def get_KNN_snapshots(target: TargetDataStruct, t,n_neighbors=5):
    class myDataset(Dataset):
        def __init__(self, target: TargetDataStruct):
            self.images_source=[(i,ti.field[0][None,:,:]) for (i,ti) in enumerate(target)]
            
        def __len__(self):
            return len(self.images_source)
        def __getitem__(self,idx):
            return self.images_source[idx]
    train_dataset = myDataset(target)
    batch_size = int(np.round(len(train_dataset)/10))
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    
    list_train_indexes =[]
    list_loss = []
    for i_batch, sample_batched in enumerate(train_dataloader):
        train_indexes, train_features = sample_batched
        target_features = t.field[0][None,None,:,:].repeat(train_features.shape[0],1,1,1)
        loss = ImagesLoss(target_features, train_features,blur=0.0001,scaling=0.9)
        list_loss.append(loss)
        list_train_indexes.append(train_indexes)
    total_train_indexes = torch.hstack(list_train_indexes)
    wasserstein_distances = torch.hstack(list_loss)
    index_sorted = torch.argsort(wasserstein_distances)
    distances_neighbors = wasserstein_distances[index_sorted[:n_neighbors]]
    index_neighbors = total_train_indexes[index_sorted[:n_neighbors]]
    return distances_neighbors, index_neighbors



def find_KNN_snapshots(snapshots, t,k=5):
    '''
    Find k nearest neighbors on snapshots space
    snapshots: 
    t:

    '''
    class myDataset(Dataset):
        def __init__(self, snapshots):
            self.images_source=[(i,ti[None,:,:]) for (i,ti) in enumerate(snapshots)]
            
        def __len__(self):
            return len(self.images_source)
        def __getitem__(self,idx):
            return self.images_source[idx]
    train_dataset = myDataset(snapshots)
    batch_size = int(np.round(len(train_dataset)/10))
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    
    list_train_indexes =[]
    list_loss = []
    for i_batch, sample_batched in enumerate(train_dataloader):
        train_indexes, train_features = sample_batched
        target_features = t[None,None,:,:].repeat(train_features.shape[0],1,1,1)
        loss = ImagesLoss(target_features, train_features,blur=0.0001,scaling=0.9)
        list_loss.append(loss)
        list_train_indexes.append(train_indexes)
    total_train_indexes = torch.hstack(list_train_indexes)
    wasserstein_distances = torch.hstack(list_loss)
    index_sorted = torch.argsort(wasserstein_distances)
    distances_neighbors = wasserstein_distances[index_sorted[:k]]
    index_neighbors = total_train_indexes[index_sorted[:k]]
    return distances_neighbors, index_neighbors




def find_knn(D_ij,k=5):
    ''' distance_matrix D is (N,N) matrix
        return (N,k)
    '''
    N = D_ij.shape[0]
    indices = []
    for i in range(N):
        index_sorted = torch.argsort(D_ij[i])
        indices.append(index_sorted[1:k+1])
    return torch.vstack(indices)

# def find_knn_ball(D_ij,eps):
#     '''
#     distnace matrix D is (N,N) matrix
#     return :
#         (N,k) array of intergers

#     '''
#     N = D_ij.shape[0]
#     # Find good k value for all data
#     list_index_sorted =[]
#     list_k =[]
#     for i in range(N):
#         index_sorted = torch.argsort(D_ij[i])
#         print('id_s',index_sorted)
#         distance_sorted = D_ij[i][index_sorted]
#         print('d_i',distance_sorted)
#         k_i=torch.sum(distance_sorted <eps)
#         print('k_i',k_i)
#         list_k.append(k_i)
#         list_index_sorted.append(index_sorted)
#     print('list_k',list_k)
#     k = min(list_k)
#     print('k',k)
#     indices = [index_sorted[1:k+1] for index_sorted in list_index_sorted]
#     # indices = []
#     # for i in range(N):
#     #     index_sorted = torch.argsort(D_ij[i])
#     #     indices.append(index_sorted[1:k+1])

#     return torch.vstack(indices)

def get_anisotropic_norm(x, U):
    """
    x is (N, D), a collection of points
    U is (N, D, D), a collection of local metrics
    
    we return an (N, N) array of distances.
    """
    N, D = x.shape

    dUd_ij = torch.zeros(N,N)
    for i in range(N):
        for j in range(N):
            diff_ij = x[i]-x[j]
            dUd_ij[i,j]= diff_ij.dot(U[i].matmul(diff_ij))

    return dUd_ij

def get_anisotropic_product(x,U,y):
    if y.dim() == 1:
        y = y.reshape(1,-1)
    Nx = x.shape[0]
    Ny = y.shape[0]
    dUd_ij = torch.zeros(Nx,Ny)
    for i in range(Nx):
        for j in range(Ny):
            diff_ij = x[i]-y[j]
            dUd_ij[i,j]= diff_ij.dot(U[i].matmul(diff_ij))
    return dUd_ij

def get_anisotropic_product_knn(x,U,y,k=5):
    if y.dim() == 1:
        y = y.reshape(1,-1)
    Nx = x.shape[0]
    Ny = y.shape[0]
    dUd_ij = torch.zeros(Nx,Ny)
    for i in range(Nx):
        for j in range(Ny):
            diff_ij = x[i]-y[j]
            dUd_ij[i,j]= diff_ij.dot(U[i].matmul(diff_ij))
    transpose_dUd_ij = torch.transpose(dUd_ij,0,1)
    sorted, indices = torch.sort(transpose_dUd_ij, dim=1)
    return indices[:,:k],sorted[:,:k]

# def get_anisotropic_product_knn(x,U,y,k=5):
#     if y.dim() == 1:
#         y = y.reshape(1,-1)
#     Nx = x.shape[0]
#     Ny = y.shape[0]
#     dUd_ij = torch.zeros(Ny,Nx)
#     for i in range(Ny):
#         for j in range(Nx):
#             diff_ij = y[i]-x[j]
#             dUd_ij[i,j]= diff_ij.dot(U[j].matmul(diff_ij))
#     #print(dUd_ij)
#     # indices_sort = torch.argsort(dUd_ij,dim=1)
#     # indices = indices_sort[:,:k]
#     # list_RdUd = []
#     # for i in range(Ny):
#     #     list_RdUd.append(dUd_ij[i][indices[i]])
#     # RdUd = torch.vstack(list_RdUd)
#     sorted, indices = torch.sort(dUd_ij, dim=1)
    
#     return indices[:,:k],sorted[:,:k]

# def anisotropic_predict(points_train, snapshots_train,U, point_test,true_target,k,
#                         params_opt = {'optimizer': 'Adam', 'lr': 0.01, 'nmax': 5, 'gamma':1, 'k_sparse':5 },
#                         params_sinkhorn_bary = {'blur': 0.001 , 'p':2, 'scaling_N':100,'backward_iterations':7}):
#     # target_train = TargetDataStruct(field=snapshots_train)
#     # D, _ = get_dissimilarity_matrix(target_train)
#     # sqdists = D.relu().sqrt()
#     # U = compute_scalings(points_train, sqdists, k=20, nits=1, robust=True, cvx=True)

#     n_keep = params_opt['k_sparse']
#     gamma = params_opt['gamma']
#     niter = params_opt['nmax']

#     def get_optimizer(weights):
#         if params_opt['optimizer'] == 'Adam':
#             return torch.optim.Adam([weights], lr=params_opt['lr'])
#         else:
#             return torch.optim.SGD([weights], lr=params_opt['lr'], momentum=0.9)
#     weights = torch.nn.Parameter(torch.ones((1,k), dtype=dtype, device=device)/k)

#     optimizer = get_optimizer(weights)
#     S = torch.arange(k,device=device)
#     evolution = {'loss':[],'support':[],'weight':[],'true_error':[]}


#     indices, RdUd = get_anisotropic_product_knn(points_train,U,point_test,k)
#     indices = indices.flatten()
#     RdUd = RdUd.flatten().to(device=device,dtype=dtype)
#     print('myindices',indices)
#     print('RdUd',RdUd)
#     alpha = torch.pow(RdUd,-1)
#     alpha = alpha/torch.sum(alpha)
#     print('alpha',alpha)
#     mse = torch.nn.MSELoss()
#     # compute the barycenter 
#     measures = torch.cat([snapshots_train[indices[id]][None,None,:,:] for id in range(k)], dim=1)
#     targets = torch.cat([snapshots_train[indices[id]][None,None,:,:] for id in range(k)], dim=0)

#     for iter in tqdm(range(niter)):
#         optimizer.zero_grad()
#         bary  = ImagesBarycenter_v2(measures=measures, weights=weights,**params_sinkhorn_bary)
#         # method 1
#         barycenters = bary.repeat(k,1,1,1)
#         distances = ImagesLoss(barycenters,targets,blur=0.001,scaling=0.8)
#         distances= distances.type(dtype)
#         #loss = mse(distances,RdUd)
#         true_error = ImagesLoss(bary,true_target[None,None,:,:],blur=0.001,scaling=0.8)
#         #loss = true_error


#         # method 2
#         loss = 0.
#         # for id in range(k):
#         #     #loss = loss + alpha[id]*(RdUd[id]-ImagesLoss(bary,snapshots_train[indices[id]][None,None,:,:],blur=0.001,scaling=0.8)) 
#         #     loss = loss + alpha[id]*(ImagesLoss(bary,snapshots_train[indices[id]][None,None,:,:],blur=0.001,scaling=0.8)) 
#         # method 3
#         for i in range(k):
#             for j in range(k):
#                 ui_uy = ImagesLoss(bary,snapshots_train[indices[i]][None,None,:,:],blur=0.001,scaling=0.8)
#                 uj_uy = ImagesLoss(bary,snapshots_train[indices[j]][None,None,:,:],blur=0.001,scaling=0.8)
#                 ui_uj = ImagesLoss(snapshots_train[indices[i]][None,None,:,:],snapshots_train[indices[j]][None,None,:,:],blur=0.001,scaling=0.8)
#                 loss = loss + (ui_uy + uj_uy -ui_uj)**2



#         loss.backward()
#         optimizer.step()
#         # keep n largest value 
#         if n_keep < k:
#             index_sort= torch.argsort(weights.data,dim=1,descending=True).flatten()
#             index_keep = index_sort[:n_keep]
#             index_zero = index_sort[n_keep:]
#             weights.data[:,index_zero]=0
            
#             # projection on the simplex
#             weights.data[:,index_keep] = mat2simplex(weights[:,index_keep]*gamma)
#         else:
#             weights.data= mat2simplex(weights*gamma)
#         S_index= torch.nonzero(torch.where(weights > 1.e-4, weights, 0.).flatten()).flatten()
#         print('Iter {}, weights {}, loss = {}, true_error = {}'.format(iter,weights,loss.item(),true_error.item()))

#         evolution['loss'].append(loss.item())
#         evolution['support'].append( S[S_index].cpu())
#         evolution['true_error'].append(true_error.item())
#     return bary, weights, evolution
    






def anisotropic_knn(x, U, k=None):
    """
    x is (N, D), a collection of points
    U is (N, D, D), a collection of local metrics
    k is an integer, the number of neighbors

    we return an (N, k) array of integers, the indices of the k-nearest neighbors.
    """
    N, D = x.shape

    # Encoding as KeOps LazyTensor:
    x_i = LazyTensor(x.view(N, 1, D))
    x_j = LazyTensor(x.view(1, N, D))
    U_i = LazyTensor(U.view(N, 1, D*D))

    # Anisotropic distances:
    diff_ij = x_j - x_i  # (N, N, D)
    dUd_ij = (diff_ij | U_i.matvecmult(diff_ij))  # (N, N, 1)

    # K-nearest neighbor search:
    indices = dUd_ij.argKmin(k + 1, dim=1)  # (N, K+1)
    # Remove "self":
    return indices[:, 1:].contiguous()

def anisotropic_product_knn(x, U, y, k=None):
    """
    x is (Nx, D), a collection of points
    U is (Nx, D, D), a collection of local metrics
    y is (Ny,D), a collection of points
    k is an integer, the number of neighbors

    we return an (Ny, k) array of integers, the indices of the k-nearest neighbors.
    """
    Nx, D = x.shape
    Ny = y.shape[0]

    # Encoding as KeOps LazyTensor:
    x_i = LazyTensor(x.view(Nx, 1, D))
    y_j = LazyTensor(y.view(1, Ny, D))
    U_i = LazyTensor(U.view(Nx, 1, D*D))

    # Anisotropic distances:
    diff_ij = y_j - x_i  # (Nx, Ny, D)
    dUd_ij = (diff_ij | U_i.matvecmult(diff_ij))  # (N, N, 1)

    # K-nearest neighbor search:
    indices = dUd_ij.argKmin(k , dim=0)  # (N, K+1)
    # Remove "self":
    return indices[:, :].contiguous()




def update_scalings(U_old, parameters, fetch_Dij, k=10, robust=True, cvx=False, cvxpylayer=None):
    """
    U_old is (N, D, D), a collection of local metrics
    parameters is (N, D), a collection of "input" parameters for the solver
    wass_sqdistances is an (N, K) -> (N, K) function
    """
    N, D = parameters.shape

    # Compute the differences to the K-NN for the metric U_old:
    #nn_indices = anisotropic_knn(parameters, U_old, k=k) # (N, K)
    nn_indices = find_knn(fetch_Dij,k=k)

    nn_params = parameters[nn_indices.view(-1), :].view(N, k, D)
    nn_diffs = nn_params - parameters.view(N, 1, D)  # (N, K, D)
    nn_sqnorms = (nn_diffs ** 2).sum(-1, keepdim=True)  # (N, K, 1)

    # Fetch the relevant Wasserstein distances:
    #nn_sqwass = fetch_Dij(nn_indices)  # (N, K)
    
    # print('Dij',fetch_Dij)
    # print('nn',nn_indices)

    nn_sqwass = fetch_dense(fetch_Dij)(nn_indices)  # (N, K)
    #mynn_sqwass = fetch_sqwass(fetch_Dij,nn_indices)
    #nn_sqwass = (nn_diffs ** 2).sum(-1)  

    nn_sqwass = nn_sqwass.view(N, k, 1)

    if cvx:
        # Rescale the differences as required - (N, K, D):
        if robust:
            nn_sourcediffs = nn_diffs / (nn_sqwass.sqrt() + 1.e-5)
            vals_tch = torch.ones(N, k).type_as(U_old)
        else:
            nn_sourcediffs = nn_diffs
            vals_tch = nn_sqwass.view(N, k)

        dxdx_tch = nn_sourcediffs.view(N, k, D, 1) * nn_sourcediffs.view(N, k, 1, D)
        dxdx_tch = dxdx_tch.view(N, k, D * D)

        # Solve the problem:
        U, = cvxpylayer(dxdx_tch, vals_tch)

    else:
        # Rescale the differences as required - (N, K, D):
        if robust:
            nn_sourcediffs = nn_diffs / (nn_sqnorms.sqrt() + 1e-4)
            nn_targetdiffs = nn_diffs * (nn_sqwass.sqrt() / (nn_sqnorms + 1e-8))
        else:
            nn_sourcediffs = nn_diffs
            nn_targetdiffs = nn_diffs * (nn_sqwass.sqrt() / (nn_sqnorms.sqrt() + 1e-4))

        # Compute the covariance matrices - (N, D, K) @ (N, K, D) = (N, D, D):
        sourcecovs = torch.bmm(nn_sourcediffs.transpose(1, 2), nn_sourcediffs)  # (N, D, D)
        targetcovs = torch.bmm(nn_targetdiffs.transpose(1, 2), nn_targetdiffs)  # (N, D, D)

        # Compute the inverse square root of the source covariances:
        L, Q = torch.linalg.eigh(sourcecovs)  # (N, D), (N, D, D)
        L = L.view(N, 1, D)
        correction = Q  / (1e-4 + L.sqrt().view(N, 1, D))  #(N, D, D)
        correction = torch.bmm(correction, Q.transpose(1, 2))  #(N, D, D)

        # Compute the local scaling as (Scov)^(-1/2) @ Tcov @ (Scov)^(-1/2)
        U = correction @ targetcovs @ correction

    return U


def compute_scalings(parameters, fetch_Dij, nits=5, k=10, cvx=True, robust=True):
    
    # Initialize with the naive Euclidean metric:
    N, D = parameters.shape
    U = torch.ones(N, 1, 1) * torch.eye(D).view(1, D, D)
    U = U.to(parameters.device)
    eta=0.000

    if cvx:
        u = cp.Variable((D, D), PSD=True)  # Constrained to be positive semi-definite
        dxdx = cp.Parameter((k, D * D))
        vals = cp.Parameter(k)

        dx_U_dx = dxdx @ cp.vec(u)  # (K,)
        dx_I_dx = dxdx @ cp.vec(torch.eye(D))
        objective = cp.Minimize(cp.pnorm(dx_U_dx + eta*dx_I_dx- vals, p=2))
        constraints = []
        problem = cp.Problem(objective, constraints)
        assert problem.is_dpp()

        cvxpylayer = CvxpyLayer(problem, parameters=[dxdx, vals], variables=[u])

    else:
        cvxpylayer = None

    # Fixed point iteration for our scaling computation:
    for it in range(nits):
        U = update_scalings(U, parameters, fetch_Dij, k=k, robust=robust, cvx=cvx, cvxpylayer=cvxpylayer)

    return U + eta*torch.eye(D)


def matrix_pow(matrix: torch.Tensor, p: float) -> torch.Tensor:
    r"""
    Power of a matrix using Eigen Decomposition.
    Args:
        matrix: matrix
        p: power
    Returns:
        Power of a matrix
    """
    #vals, vecs = torch.eig(matrix, eigenvectors=True)
    vals, vecs = torch.linalg.eig(matrix)
    #vals = torch.view_as_complex(vals.contiguous())
    vals_pow = vals.pow(p)
    vals_pow = torch.view_as_real(vals_pow)[:, 0]
    vecs = torch.real(vecs)
    matrix_pow = torch.matmul(vecs, torch.matmul(torch.diag(vals_pow), torch.inverse(vecs)))
    return matrix_pow

def affine_invariant_distance(D1,D2):
    '''
    D1: (D,D) matrix
    D2: (D,D) matrix
    Return:
        The distance between D1 and D2
    '''
    
    D = torch.matmul(matrix_pow(D1,-0.5), torch.matmul(D2,matrix_pow(D1,-0.5)))
    S = torch.linalg.svdvals(D)
    #N2 = torch.pow(torch.log(S),2).sum()
    
    N2 = S.log().pow(2).sum()
    
    return N2
def get_affine_invariant_distance_matrix(U):
    '''
    U : (N,D,D)
    return:
        tensor (N,N)
    '''
    N= U.shape[0]

    dU = torch.zeros(N,N)
    for i in range(N):
        for j in range(N):
            dU[i,j]= affine_invariant_distance(U[i],U[j])

    return dU



def get_distance_matrix(s1,s2):

    class PairWiseDataset(Dataset):
        def __init__(self, s1,s2):
            self.images_source=[]
            self.images_target=[]
            for ((i,ti), (j,tj)) in itertools.product(enumerate(s1), enumerate(s2)):
                self.images_source.append((i,ti[None,:,:]))
                self.images_target.append((j,tj[None,:,:]))

        def __len__(self):
            return len(self.images_source)
        def __getitem__(self,idx):
            return self.images_source[idx], self.images_target[idx]

    myDataset = PairWiseDataset(s1,s2)

    Total_loss = []
    firstIndex =[]
    secondIndex=[]
    train_dataloader = DataLoader(myDataset, batch_size=100, shuffle=False)
    #train_features, train_labels = next(iter(train_dataloader))
    for i_batch, sample_batched in enumerate(train_dataloader):
        train_features, train_labels = sample_batched
        loss = ImagesLoss(train_features[1], train_labels[1],blur=0.0001,scaling=0.9)
        Total_loss.append(loss)
        firstIndex.append(train_features[0])
        secondIndex.append(train_labels[0])
    wasserstein_distances = torch.hstack(Total_loss)
    wasserstein_firstIndex   = torch.hstack(firstIndex)
    wasserstein_secondIndex = torch.hstack(secondIndex)

    Nx = len(s1)
    Ny = len(s2)
    distance_matrix = torch.zeros(Nx, Ny)
    
    for  id in range(wasserstein_distances.shape[0]):
        distance_matrix[wasserstein_firstIndex[id]][wasserstein_secondIndex[id]] = wasserstein_distances[id]
        
    return distance_matrix

def barycentric_coordinates(X,x):
    """Only valid when X is triangle"""

    xlast = X[-1]
    dX = [ xi - xlast for xi in X]
    T = np.vstack(dX[:-1]).T
    λ = np.linalg.solve(T, x-xlast)
    λ_last = 1 - np.sum(λ)
    return np.array([*λ, λ_last])


def best_weight_Dirac_masses(main_points,x,params_opt = {'optimizer': 'Adam', 'lr': 0.001, 'nmax': 5, 'gamma':1, 'k_sparse':5 }):

    N = main_points.shape[0]
    #n_keep = params_opt['k_sparse']
    gamma = params_opt['gamma']
    niter = params_opt['nmax']
    def get_optimizer(weights):
        if params_opt['optimizer'] == 'Adam':
            return torch.optim.Adam([weights], lr=params_opt['lr'])
        else:
            return torch.optim.SGD([weights], lr=params_opt['lr'], momentum=0.9)
    weights = torch.nn.Parameter(torch.tensor(np.array([[0.1, 0.8, 0.1]]), dtype=dtype, device=device))

    optimizer = get_optimizer(weights)
    S = torch.arange(N,device=device)
    evolution = {'loss':[],'support':[],'weight':[],'true_error':[]}
    mse = torch.nn.MSELoss()
    for iter in tqdm(range(niter)):
        evolution['weight'].append(weights.detach().clone())
        optimizer.zero_grad()
        bary = torch.matmul(weights,main_points)
        loss = N*mse(torch.norm(x-main_points,dim=1).pow(2), torch.norm(bary-main_points,dim=1).pow(2))
        loss.backward()
        optimizer.step() 

        weights.data= mat2simplex(weights*gamma)
        S_index= torch.nonzero(torch.where(weights > 1.e-4, weights, 0.).flatten()).flatten()
        print('Iter {}, weights {}, loss = {}'.format(iter,weights,loss.item()))
    
        evolution['loss'].append(loss.item())
        evolution['support'].append( S[S_index].cpu())
    return bary, weights, evolution


