# SPDX-FileCopyrightText: The sinkhorn-rom authors, see file AUTHORS
#
# SPDX-License-Identifier: GPL-3.0-or-later

import torch
import itertools
import scipy.interpolate
import numpy as np
from sklearn.neighbors import NearestNeighbors
import uuid

from .BaseModel import BaseModel
from ..DataManipulators.DataStruct import DataStruct, TargetDataStruct, QueryDataStruct
from ..Evaluators.Barycenter import ImagesLoss, projGraFixSupp, projGraAdapSupp, projRGraSP, ImagesBarycenter_v2
from ...utils import timeit, check_create_dir, pprint
from ...config import fit_dir, dtype, device, results_dir
from ...tools import get_dissimilarity_matrix, compute_scalings, get_anisotropic_norm, get_anisotropic_product_knn, find_KNN_snapshots, get_anisotropic_product, get_distance_matrix
from ...visualization import plot_fields_images
import pickle
from tqdm import tqdm
import pandas as pd

from torch.utils.data import Dataset, DataLoader 
import seaborn as sns
import  matplotlib.pyplot as plt

class Embedding(BaseModel):
    def __init__(self,
                #params_regression = {'type_regression':'Ridge', 'reg_param': 5.e-3,'n_neighbors':2, 'adaptive_knn':False},
                #params_kernel       = {'metric':'rbf', 'gamma':1},
                params_embedding = {'nits': 1,'k':100, 'robust':True, 'cvx':True},
                params_online = {'n_neighbors':10, 'method':'IWD'},
                params_sinkhorn_bary = {'blur': 0.001 , 'p':2, 'scaling_N':100,'backward_iterations':5},
                params_opt_best_barycenter = {'optimizer': 'Adam','lr': 0.1,'nmax': 100,'type_loss':'W2','gamma':1,'k_sparse':10},
                name='Embedding'):
        """

        @param fit_parameter_range: which values to find optimal value of RKHS parameter
        @param kernel_family: gaussian, exponential, this gives the appropiate kernel function to be used
        @param metric: metric to evaluate the performance in the optimization
        """
        super().__init__(name=name)

        #self.uuid_training = uuid.uuid4()

        self.interpolator_tools = []
        
        # Sinkhorn parameters
        #self.params_regression = params_regression
        #self.params_kernel     = params_kernel
        self.params_embedding = params_embedding
        self.params_online = params_online
        self.params_sinkhorn_bary = params_sinkhorn_bary
        self.params_opt_best_barycenter = params_opt_best_barycenter


    def fit(self, query: QueryDataStruct, target: TargetDataStruct):
        """
        @param query: QueryDataStruct with information about
            * parameters: paramter values
            * field_spatial_coordinates: coordinates where we want to infer field values. (n_coords_infer, spatial_dim)
        @param target: TargetDataStruct with the information of the field values for each parameter and for the
        field_spatial_coordinates.
        @return:
        """   
        with timeit('Fit with Embedding algorithm', font_style='bold', bg='Red', fg='White'):
            n_samples = len(query)
            self.interpolator_tools = self.prepare_interpolation_tools(query, target)
            




    def prepare_interpolation_tools(self, query: QueryDataStruct, target: TargetDataStruct):
        interpolator_tools = {'parameters': None, 'snapshots': None, 'U':None,'D':None}
        with timeit('Prepare local interpolation tools', font_style='bold'):
            interpolator_tools['parameters'] = torch.Tensor(np.array([param.tolist() for param in query.parameter])) # Each line contains the parameters of a function from query
            interpolator_tools['snapshots'] = torch.Tensor(np.array([u.field[0].cpu().numpy() for u in target]))
            
            with timeit('Prepare dissimilarity matrix', font_style='bold'):
                D, _ = get_dissimilarity_matrix(target)
            sqdists = D.relu()
            points = interpolator_tools['parameters']
            with timeit('Prepare local Embedding matrix', font_style='bold'):
                U = compute_scalings(points, sqdists, **self.params_embedding)

            #dUd = get_anisotropic_norm(points,U)
            # fig =sns.jointplot(x=dUd.sqrt().view(-1), y=D.sqrt().view(-1))
            # plt.show()

            interpolator_tools['U'] = U.cpu()
            interpolator_tools['D'] = D.cpu()
            return interpolator_tools


    # def prepare_dataloader(self,target: TargetDataStruct):
    #     # Create dataset for training
    #     class myDataset(Dataset):
    #         def __init__(self, target: TargetDataStruct):
    #             self.images_source=[(i,ti.field[0][None,:,:]) for (i,ti) in enumerate(target)]
                
    #         def __len__(self):
    #             return len(self.images_source)
    #         def __getitem__(self,idx):
    #             return self.images_source[idx]
    #     train_dataset = myDataset(target)
    #     #batch_size = int(np.round(len(train_dataset)/10))
    #     batch_size = 32
    #     self.train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
        

    def save_fit(self, rootdir):
        """Save fit to load afterwards
        """
        # From tensor to numpy
        interpolator_tools = {'parameters': None, 'snapshots': None, 'U':None,'D':None}
        interpolator_tools['parameters'] = self.interpolator_tools['parameters'].cpu().numpy()
        interpolator_tools['snapshots'] = self.interpolator_tools['snapshots'].cpu().numpy()
        interpolator_tools['U'] = self.interpolator_tools['U'].cpu().numpy()
        interpolator_tools['D'] = self.interpolator_tools['D'].cpu().numpy()

        pickle.dump(interpolator_tools, open(rootdir+'interpolator_tools', "wb"))


    def load_fit(self, rootdir):
        """Load fit to be able to call the predict method
        """
        self.interpolator_tools = pickle.load(open(rootdir+'interpolator_tools', "rb"))
        # From numy to tensor
        self.interpolator_tools['parameters'] = torch.tensor(self.interpolator_tools['parameters'],dtype=dtype)
        self.interpolator_tools['snapshots'] = torch.tensor(self.interpolator_tools['snapshots'],dtype=dtype)
        self.interpolator_tools['U'] = torch.tensor(self.interpolator_tools['U'],dtype=dtype)
        self.interpolator_tools['D'] = torch.tensor(self.interpolator_tools['D'],dtype=dtype)
        #self.interpolator_tools['dUd'] = torch.tensor(self.interpolator_tools['dUd'],dtype=dtype)






    def weights_core_predict(self, q:QueryDataStruct,target:TargetDataStruct):
        """
        @param query: QueryDataStruct with information about
            * parameters: paramter values
            * field_spatial_coordinates: coordinates where we want to infer field values. (n_coords_infer, spatial_dim)
        @return: predicted field at field_spatial_coordinates
        """
        with timeit('Core predict', font_style='bold', bg='Red', fg='White'):

            root_dir = check_create_dir(results_dir+'/embedding_core/')
            k = self.params_online['n_neighbors']

            point_test = q.parameter[0].to(dtype=dtype,device=torch.device('cpu'))

            points_train = self.interpolator_tools['parameters'].to(dtype=dtype)
            U = self.interpolator_tools['U'].to(dtype=dtype)
            snapshots_train = self.interpolator_tools['snapshots'].to(dtype=dtype,device=device)

            #snapshots_train = torch.tensor(self.interpolator_tools['snapshots'], device=device)
            #snapshots_train = torch.from_numpy(self.interpolator_tools['snapshots']).to(device=device)
            # check quality of matrix U
            # dUd=get_anisotropic_product(points_train,U,point_test)
            # snapshot_test = [target.field[0]]
            # D = get_distance_matrix(snapshots_train,snapshot_test)
            # fig =sns.jointplot(x=dUd.view(-1), y=D.view(-1))
            # plt.show()

            # indices, RdUd = get_anisotropic_product_knn(points_train,U,point_test,k=k)
            # indices = indices.flatten()
            # RdUd = RdUd.flatten().to(device=device,dtype=dtype)
            # measures = torch.cat([snapshots_train[indices[id]][None,None,:,:] for id in range(k)], dim=1)
            # target_field = target.field[0][None,None,:,:]
            # total_measures = torch.cat([snapshots_train[indices[id]][None,None,:,:] for id in range(k)], dim=0)
            # targets =  target_field.repeat(k, 1, 1, 1).to(dtype=dtype)
            # distance_true = ImagesLoss(targets,total_measures,blur=0.001,scaling=0.8)
            # print('distance_true : ',distance_true)



            # distance_neighbors,index_neighbors = find_KNN_snapshots(snapshots_train, target.field[0],k=k)
            # index_neighbors = index_neighbors.flatten()
            # print('indexN',index_neighbors)
            # print('distanceN',distance_neighbors)




            # Inverse weights distance
            # alpha = torch.pow(RdUd,-1)
            # alpha = alpha/torch.sum(alpha)
            # weight = alpha[None,:].to(device)
            # barycenter  = ImagesBarycenter_v2(measures=measures, weights=weight,**self.params_sinkhorn_bary)

            # Best barycenter
            if self.params_online['method'] == 'AS':

                indices, RdUd = get_anisotropic_product_knn(points_train,U,point_test,k=k)
                indices = indices.flatten()
                RdUd = RdUd.flatten().to(device=device,dtype=dtype)
                measures = torch.cat([snapshots_train[indices[id]][None,None,:,:] for id in range(k)], dim=1)
                target_field = target.field[0][None,None,:,:]

                barycenter, weight, evolution = projGraAdapSupp(target_field= target_field,measures = measures,
                                field_coordinates=q.field_spatial_coordinates[0],
                                params_opt = self.params_opt_best_barycenter,
                                params_sinkhorn_bary=self.params_sinkhorn_bary,
                                distance_target_measures=RdUd)
                result = {'barycenter':barycenter[0,0], 'weight':weight[0],'loss':evolution['loss'][-1], 'index_KNN': indices}

            elif self.params_online['method'] == 'AS_Bench':

                #distance_neighbors,indices = find_KNN_snapshots(snapshots_train, target.field[0],k=k)
                indices, RdUd = get_anisotropic_product_knn(points_train,U,point_test,k=k)
                RdUd = RdUd.flatten().to(device=device,dtype=dtype)
                indices = indices.flatten()
                
                measures = torch.cat([snapshots_train[indices[id]][None,None,:,:] for id in range(k)], dim=1)
                target_field = target.field[0][None,None,:,:]

                total_measures = torch.cat([snapshots_train[indices[id]][None,None,:,:] for id in range(k)], dim=0)
                targets =  target_field.repeat(k, 1, 1, 1).to(dtype=dtype)
                distance_true = ImagesLoss(targets,total_measures,blur=0.001,scaling=0.8)
                
                barycenter, weight, evolution = projGraAdapSupp(target_field= target_field,measures = measures,
                                field_coordinates=q.field_spatial_coordinates[0],
                                params_opt = self.params_opt_best_barycenter,
                                params_sinkhorn_bary=self.params_sinkhorn_bary,
                                distance_target_measures=distance_true)
                result = {'barycenter':barycenter[0,0], 'weight':weight[0],'loss':evolution['loss'][-1], 'index_KNN': indices}

            elif self.params_online['method'] == 'IDW':
                indices, RdUd = get_anisotropic_product_knn(points_train,U,point_test,k=k)
                indices = indices.flatten()
                RdUd = RdUd.flatten().to(device=device,dtype=dtype)
                measures = torch.cat([snapshots_train[indices[id]][None,None,:,:] for id in range(k)], dim=1)

                #Inverse weights distance
                distances = RdUd
                min_indices = torch.argmin(distances, keepdims=False)
                idx = min_indices.item()
                #print('idx',idx)
                min_val = distances[idx]
                if min_val <= 1.e-5:
                    print('close point')
                    alpha = torch.zeros(distances.shape[0])
                    alpha[idx] = 1
                else:
                    alpha = torch.pow(distances,-1)
                    alpha = alpha/torch.sum(alpha)

                weight = alpha[None,:].to(device)
                barycenter  = ImagesBarycenter_v2(measures=measures, weights=weight,**self.params_sinkhorn_bary)
                result = {'barycenter':barycenter[0,0], 'weight':weight[0],'loss':0, 'index_KNN': indices}
            elif self.params_online['method'] == 'Nadaraya':
                print('Nadaraya with k=',k)
                indices, RdUd = get_anisotropic_product_knn(points_train,U,point_test,k=k)
                indices = indices.flatten()
                RdUd = RdUd.flatten().to(device=device,dtype=dtype)
                measures = torch.cat([snapshots_train[indices[id]][None,None,:,:] for id in range(k)], dim=1)
                # Nararaya
                distances = RdUd
                gamma=1
                alpha = torch.exp(-gamma*distances)
                alpha = alpha/torch.sum(alpha)
                weight = alpha[None,:].to(device)
                barycenter  = ImagesBarycenter_v2(measures=measures, weights=weight,**self.params_sinkhorn_bary)
                result = {'barycenter':barycenter[0,0], 'weight':weight[0],'loss':0, 'index_KNN': indices}

            else:
                print('Use the nearest neighbor !')
                indices, RdUd = get_anisotropic_product_knn(points_train,U,point_test,k=1)
                indices = indices.flatten()
                print('indices',indices)
                result = {'barycenter': snapshots_train[indices[0]], 'weight':torch.ones(1, device=device),'loss':0, 'index_KNN': indices}

        return result
        
        


    