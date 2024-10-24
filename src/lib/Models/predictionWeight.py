# SPDX-FileCopyrightText: The sinkhorn-rom authors, see file AUTHORS
#
# SPDX-License-Identifier: GPL-3.0-or-later

import torch
import itertools
import scipy.interpolate
import numpy as np
from sklearn.neighbors import NearestNeighbors
import uuid
from sklearn import manifold, datasets
from .BaseModel import BaseModel
from ..DataManipulators.DataStruct import DataStruct, TargetDataStruct, QueryDataStruct
from ..Evaluators.Barycenter import ImagesLoss, projGraFixSupp, projGraAdapSupp, projRGraSP
from ...utils import timeit, check_create_dir, pprint
from ...config import fit_dir, dtype, device
from geomloss import SamplesLoss
import pickle
from tqdm import tqdm
import pandas as pd
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
from typing import List
from sklearn.metrics.pairwise import rbf_kernel, pairwise_kernels
from geomloss import ImagesBarycenter
import time
from sklearn import preprocessing
from torch.utils.data import Dataset, DataLoader 



class predictionWeight(BaseModel):
    def __init__(self,
                #params_regression = {'type_regression':'Ridge', 'reg_param': 5.e-3,'n_neighbors':2, 'adaptive_knn':False},
                #params_kernel       = {'metric':'rbf', 'gamma':1},
                params_KNN = {'n_neighbors':10},
                params_sinkhorn_bary = {'blur': 0.001 , 'p':2, 'scaling_N':100,'backward_iterations':5},
                params_opt_best_barycenter = {'optimizer': 'Adam','lr': 0.1,'nmax': 100,'gamma':1,'k_sparse':10},
                name='predictionWeight'):
        """

        @param fit_parameter_range: which values to find optimal value of RKHS parameter
        @param kernel_family: gaussian, exponential, this gives the appropiate kernel function to be used
        @param metric: metric to evaluate the performance in the optimization
        """
        super().__init__(name=name)

        self.uuid_training = uuid.uuid4()

        self.interpolator_tools = []
        
        # Sinkhorn parameters
        #self.params_regression = params_regression
        #self.params_kernel     = params_kernel
        self.params_KNN = params_KNN
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
        with timeit('Fit with Knn prediction algorithm', font_style='bold', bg='Red', fg='White'):
            n_samples = len(query)
            self.interpolator_tools = self.prepare_interpolation_tools(query, target)
            




    def prepare_interpolation_tools(self, query: QueryDataStruct, target: TargetDataStruct):
        interpolator_tools = {'parameters': None, 'nn_param': None, 'target_train': None}
        with timeit('Prepare local interpolation tools', font_style='bold'):
            interpolator_tools['parameters'] = np.array([param.tolist() for param in query.parameter]) # Each line contains the parameters of a function from query
            #interpolator_tools['nn_param'] = NearestNeighbors(n_neighbors= self.params_regression['n_neighbors'], algorithm='ball_tree').fit(interpolator_tools['parameters'])
            interpolator_tools['target_train'] = np.array([u.field[0].cpu().numpy() for u in target])
            return interpolator_tools


    def prepare_dataloader(self,target: TargetDataStruct):
        # Create dataset for training
        class myDataset(Dataset):
            def __init__(self, target: TargetDataStruct):
                self.images_source=[(i,ti.field[0][None,:,:]) for (i,ti) in enumerate(target)]
                
            def __len__(self):
                return len(self.images_source)
            def __getitem__(self,idx):
                return self.images_source[idx]
        train_dataset = myDataset(target)
        #batch_size = int(np.round(len(train_dataset)/10))
        batch_size = 32
        self.train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
        

    def save_fit(self, rootdir):
        """Save fit to load afterwards
        """
        pickle.dump(self.interpolator_tools, open(rootdir+'interpolator_tools', "wb"))


    def load_fit(self, rootdir):
        """Load fit to be able to call the predict method
        """
        self.interpolator_tools = pickle.load(open(rootdir+'interpolator_tools', "rb"))
        snapshots = torch.tensor(self.interpolator_tools['target_train']).to(device)
        target = TargetDataStruct(snapshots)
        self.prepare_dataloader(target)






    def weights_core_predict(self, q:QueryDataStruct,target:TargetDataStruct):
        """
        @param query: QueryDataStruct with information about
            * parameters: paramter values
            * field_spatial_coordinates: coordinates where we want to infer field values. (n_coords_infer, spatial_dim)
        @return: predicted field at field_spatial_coordinates
        """
        

        n_train = self.interpolator_tools['target_train'].shape[0]
        N = self.params_KNN['n_neighbors']
        if N < n_train:
            print('KNN')
            list_indexes =[]
            list_loss = []
            for i_batch, sample_batched in enumerate(self.train_dataloader):
                train_indexes, train_features = sample_batched
                target_features = target.field[0][None,None,:,:].repeat(train_features.shape[0],1,1,1)
                loss = ImagesLoss(target_features, train_features,blur=0.0001,scaling=0.9)
                list_loss.append(loss)
                list_indexes.append(train_indexes)
            total_indexes = torch.hstack(list_indexes)
            total_distances = torch.hstack(list_loss)
        
            index_sorted = torch.argsort(total_distances)
            # total_distances_sorted = total_distances[index_sorted]
            # total_indexes_sorted   = total_indexes[index_sorted]
            index_neighbors = total_indexes[index_sorted[:N]]
            fields = [torch.tensor(self.interpolator_tools['target_train'][i], device=device) for i in index_neighbors.tolist()]

        else:
            print('Using  all training snapshots !')
            index_neighbors = torch.arange(n_train)
            fields = [torch.tensor(self.interpolator_tools['target_train'][i], device=device) for i in range(n_train)]


        measures = torch.cat([field[None,None,:,:] for field in fields], dim=1)
        barycenter, weight, evolution = projGraAdapSupp(target_field=target.field[0][None,None,:,:],measures = measures,
                        field_coordinates=q.field_spatial_coordinates[0],
                        params_opt = self.params_opt_best_barycenter,
                        params_sinkhorn_bary=self.params_sinkhorn_bary)
        #print('index_neighbors',index_neighbors)
        result = {'barycenter':barycenter[0,0], 'weight':weight[0], 'loss':evolution['loss'][-1], 'index_KNN': index_neighbors}
        return result
        #return barycenter[0,0], weight[0], evolution['loss'][-1], index_neighbors


    