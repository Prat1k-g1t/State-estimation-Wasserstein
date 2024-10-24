# SPDX-FileCopyrightText: The sinkhorn-rom authors, see file AUTHORS
#
# SPDX-License-Identifier: GPL-3.0-or-later

import torch
import itertools
import scipy.interpolate
import numpy as np
from sklearn.neighbors import NearestNeighbors
import uuid
from natsort import natsorted
from .BaseModel import BaseModel
from ..DataManipulators.DataStruct import DataStruct, TargetDataStruct, QueryDataStruct
from ..Evaluators.Barycenter import  ImagesLoss, ImagesBarycenter_v2, \
    mat2simplex, projGraFixSupp, projGraAdapSupp, ImagesBarycenter_1d
from ...utils import timeit, check_create_dir, pprint
from ...config import fit_dir, dtype, device
from torch.utils.data import Dataset, DataLoader

from geomloss import SamplesLoss
from geomloss.utils import dimension

import matplotlib.pyplot as plt

class NonIntrusiveGreedyImages(BaseModel):
    def __init__(self, Loss, nmax=np.Inf,
                compute_intermediate_interpolators = True,
                params_sinkhorn_bary = {'blur': 0., 'p':2, 'scaling_N':200,'backward_iterations':5} ,
                params_opt_best_barycenter = {'optimizer': 'Adam','lr': 0.01,'nmax': 50,'gamma':1,'k_sparse':5},
                name='NonIntrusiveGreedyImages'):
        """

        @param fit_parameter_range: which values to find optimal value of RKHS parameter
        @param kernel_family: gaussian, exponential, this gives the appropiate kernel function to be used
        @param metric: metric to evaluate the performance in the optimization
        """
        super().__init__(name=name)

        # Loss
        self.Loss = Loss

        # Fitting
        self.nmax = nmax
        self.fit_output = None
        self.fit_conv = None
        self.index_fit_fields = None
        self.final_barycenter = None
        self.uuid_training = uuid.uuid4()
        self.compute_intermediate_interpolators = compute_intermediate_interpolators

        # Interpolation of barycentric weights (list filled during the fitting for each dimension)
        self.interpolator_tools = []

        # Sinkhorn parameters
        self.params_sinkhorn_bary = params_sinkhorn_bary
        self.params_opt_best_barycenter = params_opt_best_barycenter
        #print(self.params_sinkhorn_bary)
        
    def visualization(self, x, U, measures, field):
        # measures = measures.squeeze()
        x = x.numpy()
        fig, axs = plt.subplots(3, sharex=True, sharey=True)
        # fig = plt.figure()
        # plt.plot(x.numpy().squeeze(),U.detach().numpy().squeeze())
        
        
        for i in range(len(self.fit_output)): axs[0].plot(x, measures[i,:].numpy().T)
        axs[0].set_xlabel('x')
        axs[0].set_ylabel(r'$\mu_{normalized}$', rotation=90)

        axs[1].plot(x, U.detach().squeeze().numpy().T)
        # axs[1].plot(x, bestbary.detach().squeeze().numpy().T)
        
        axs[1].legend([r'Ref. barycenter', 'best barycenter'],loc='best')
        axs[2].plot(x, field.numpy().T)
        axs[2].set_xlabel('x')
        axs[2].legend([r't.field'],loc='best')
        
        
        plt.show()


    def fit(self, query: QueryDataStruct, target: TargetDataStruct):
        """

        @param query: QueryDataStruct with information about
            * parameters: paramter values
            * field_spatial_coordinates: coordinates where we want to infer field values. (n_coords_infer, spatial_dim)
        @param target: TargetDataStruct with the information of the field values for each parameter and for the
        field_spatial_coordinates.
        @return:
        """   
        loss_samples = SamplesLoss("sinkhorn", p=2, blur = 1e-3, scaling = 0.9, debias=True)
        
        with timeit('Fit with greedy algorithm', font_style='bold', bg='Red', fg='White'):

            #dtype = target.field[0].dtype
            #dtype = dtype
            self.fit_output = list()
            self.fit_conv = list()
            self.fit_av_err = list()
            self.index_fit_fields = list()
            self.iter = list()
            
            #DEBUG:
            D = target[0].field[0].dim()
            # print('len target',len(target))

            n_effective = np.min([self.nmax, len(query)])
            # n_effective = self.nmax
            #DEBUG:
            # print('n_effective',n_effective)
            # print('nmax',self.nmax)
            for n in range(n_effective):
                
                with timeit('Iteration n={}'.format(n), font_style='bold'):
                    max_loss = 0.
                    if n==0:                    
                        # for ((i,ti), (j,tj)) in itertools.combinations(enumerate(target), 2):
                        #     #loss = self.Loss(ti.field[0].flatten(), query.field_spatial_coordinates[0], tj.field[0].flatten(), query.field_spatial_coordinates[0]).item()
                        #     loss = ImagesLoss(ti.field[0][None,None,:,:], tj.field[0][None,None,:,:],blur=0.0001,scaling=0.9).item()
                        #     if loss > max_loss:
                        #         u1, u2 = ti.field[0], tj.field[0]
                        #         i1, i2 = i, j
                        #         max_loss = loss


                        i1,i2,max_loss = self.prepare_pairwise_distances(target)
                        


                        u1 = DataStruct(parameter=query[i1].parameter,
                                        field_spatial_coordinates=query[i1].field_spatial_coordinates,
                                        field=target[i1].field)
                        u2 = DataStruct(parameter=query[i2].parameter,
                                        field_spatial_coordinates=query[i2].field_spatial_coordinates,
                                        field=target[i2].field)
                        self.fit_output.extend([u1, u2])
                        self.index_fit_fields.extend([i1, i2])
                        self.fit_conv.append(max_loss)

                        if self.compute_intermediate_interpolators or (n==n_effective-1):
                            # Compute barycentric local interpolation tools
                            n_params = query.parameter[0].shape[0]
                            n_neighbors = n_params + 3 # TODO: Consider n_neighbors as a hyperparameter
                            self.interpolator_tools.append(self.prepare_barycentric_interpolation_tools(query, target, n_neighbors))
                    else:
                        for (i, t) in enumerate(target):
                            tmp = list()
                            if i not in self.index_fit_fields:
                                if D == 1:
                                    tgt = t.field[0][:]/t.field[0][:].sum()
                                    for u in self.fit_output:
                                        tmp.append(u.field[0][:]/u.field[0][:].sum())
                                    bary, _ , _ = projGraAdapSupp(
                                                    target_field=tgt[None, None, :],
                                                    measures= torch.cat([p[None,None,:] for p in tmp], dim =1),
                                                    field_coordinates=query.field_spatial_coordinates[0],
                                                    params_opt=self.params_opt_best_barycenter,
                                                    params_sinkhorn_bary= self.params_sinkhorn_bary)
                                    
                                    # bary, _ , _ = projGraFixSupp(
                                    #                 target_field=t.field[0][None, None, :],
                                    #                 measures= torch.cat([u.field[0][None,None,:] for u in self.fit_output], dim =1),
                                    #                 field_coordinates=query.field_spatial_coordinates[0],
                                    #                 params_opt=self.params_opt_best_barycenter,
                                    #                 params_sinkhorn_bary= self.params_sinkhorn_bary)
                                    #DEBUG:
                                    loss =  ImagesLoss(tgt[None,None,:], bary, scaling=0.9).item()
                                    # loss =  loss_samples(tgt[None,None,:], bary).item()
                                    # loss =  loss_samples(t.field[0][None,None,:], bary).item()
                                    # print('Mass of barycenter',bary.sum())
                                    # print('Barycenter wts',wts)
                                else:
                                    bary, _ , _ = projGraFixSupp(
                                                    target_field=t.field[0][None,None,:,:],
                                                    measures= torch.cat([u.field[0][None,None,:,:] for u in self.fit_output], dim =1),
                                                    field_coordinates=query.field_spatial_coordinates[0],
                                                    params_opt=self.params_opt_best_barycenter,
                                                    params_sinkhorn_bary= self.params_sinkhorn_bary
                                                    )
                                    #loss = self.Loss(t.field[0].flatten(), query.field_spatial_coordinates[0], bary.flatten(), query.field_spatial_coordinates[0]).item()
                                    loss =  ImagesLoss(t.field[0][None,None,:,:], bary,blur=0.0001,scaling=0.9).item()
                                    
                                if loss > max_loss:
                                    u = t.field[0]
                                    idx = i
                                    max_loss = loss
                                    self.final_barycenter = bary
                                    
                                # if max_loss >= 1e-6: break
                            
                        un = DataStruct(parameter=query[idx].parameter,
                                        field_spatial_coordinates=query[idx].field_spatial_coordinates,
                                        field=target[idx].field)
                        self.fit_output.append(un)
                        self.index_fit_fields.append(idx)
                        self.fit_conv.append(max_loss)
                        # self.fit_av_err.append(err)
                        #DEBUG:
                        print('len fit_output',len(self.fit_output))

                        if self.compute_intermediate_interpolators or (n==n_effective-1):
                            n_params = query.parameter[0].shape[0]
                            n_neighbors = n_params + 3 # TODO: Consider n_neighbors as a hyperparameter
                            self.interpolator_tools.append(self.prepare_barycentric_interpolation_tools(query, target, n_neighbors))

    def prepare_barycentric_interpolation_tools(self, query: QueryDataStruct, target: TargetDataStruct, n_neighbors: int):
        D = target[0].field[0].dim()
        #print(self.params_sinkhorn_bary)
        barycentric_interpolator_tools = {'parameters': None, 'weights': None, 'nn': None}
        with timeit('Prepare barycentric local interpolation tools', font_style='bold'):
            # Store weights of best barycenter for each query
            barycentric_interpolator_tools['weights'] = list()
            for (i, t) in enumerate(target):
                tmp  = list()
                if i in self.index_fit_fields:
                    # If index belongs to a function selected by the greedy algorithm, its best barycentric weigts are trivial
                    barycentric_interpolator_tools['weights'].append([1. if j==self.index_fit_fields.index(i) else 0. for j in range(len(self.index_fit_fields))])
                else:
                    if D == 1:
                        tgt = t.field[0]/t.field[0].sum()
                        for u in self.fit_output:
                            tmp.append(u.field[0][:]/u.field[0][:].sum())
                        bary, weights , _ = projGraAdapSupp(
                                        target_field=tgt[None, None, :],
                                        measures= torch.cat([p[None,None,:] for p in tmp], dim =1),
                                        field_coordinates=query.field_spatial_coordinates[0],
                                        params_opt=self.params_opt_best_barycenter,
                                        params_sinkhorn_bary= self.params_sinkhorn_bary)
                        
                        # bary, weights , _ = projGraFixSupp(
                        #                             target_field=t.field[0][None,None,:],
                        #                             measures= torch.cat([u.field[0][None,None,:] for u in self.fit_output], dim =1),
                        #                             field_coordinates=query.field_spatial_coordinates[0],
                        #                             params_opt=self.params_opt_best_barycenter,
                        #                             params_sinkhorn_bary= self.params_sinkhorn_bary)
                                                    
                    else:
                        bary, weights , _ = projGraFixSupp(
                                                    target_field=t.field[0][None,None,:,:],
                                                    measures= torch.cat([u.field[0][None,None,:,:] for u in self.fit_output], dim =1),
                                                    field_coordinates=query.field_spatial_coordinates[0],
                                                    params_opt=self.params_opt_best_barycenter,
                                                    params_sinkhorn_bary= self.params_sinkhorn_bary
                                                    )
                    barycentric_interpolator_tools['weights'].append(weights.flatten().tolist())
            barycentric_interpolator_tools['weights'] = np.array(barycentric_interpolator_tools['weights']) # Each line contains the barycentric weights of a function from query
            barycentric_interpolator_tools['parameters']= np.array([param.tolist() for param in query.parameter]) # Each line contains the parameters of a function from query

            barycentric_interpolator_tools['nn'] = NearestNeighbors(n_neighbors=n_neighbors, algorithm='ball_tree').fit(barycentric_interpolator_tools['parameters'])
            return barycentric_interpolator_tools

    def prepare_pairwise_distances(self,target:TargetDataStruct):
        """
        Prepare distances
        """
        D = target[0].field[0].dim()
        
        #DEBUG:
        # print("Shape of target and field_coord", target.shape, field_coord.shape)
        
        class PairWiseDataset(Dataset):
            def __init__(self, target: TargetDataStruct):
                self.images_source=[]
                self.images_target=[]
                for ((i,ti), (j,tj)) in itertools.combinations(enumerate(target), 2):
                    #TODO: Implement a better version for 1D/multiD
                    if D == 1:
                        tmp1 = ti.field[0]/ti.field[0].sum()
                        tmp2 = tj.field[0]/tj.field[0].sum()
                        self.images_source.append((i,tmp1[None, :]))
                        self.images_target.append((j,tmp2[None, :]))
                        # self.images_source.append((i,ti.field[0][None,:]))
                        # self.images_target.append((j,tj.field[0][None,:]))
                    else:
                        self.images_source.append((i,ti.field[0][None,:,:]))
                        self.images_target.append((j,tj.field[0][None,:,:]))
            def __len__(self):
                return len(self.images_source)
            def __getitem__(self,idx):
                return self.images_source[idx], self.images_target[idx]

        myDataset = PairWiseDataset(target)
        #DEBUG:
        # print('Size of images_source',len(myDataset))
        Total_loss = []
        firstIndex =[]
        secondIndex=[]
        train_dataloader = DataLoader(myDataset, batch_size=100, shuffle=False)
        # print('Type of train_dataloader', type(train_dataloader))
        # print('Len of train_dataloader', len(train_dataloader))

        for i_batch, sample_batched in enumerate(train_dataloader):
            train_features, train_labels = sample_batched
            #DEBUG:
            # print('i_batch',i_batch)
            # print('Type of train_features, train_labels', type(train_features),type(train_labels))
            # print('Len of train_features, train_labels', len(train_features),len(train_labels))
            # print("Shape of train features[0] and labels[0]", train_features[0].shape, train_labels[0].shape)
            # print("Shape of train features[1] and labels[1]", train_features[1].shape, train_labels[1].shape)
            # loss = ImagesLoss(train_features[1], train_labels[1],scaling=0.9)
            
            loss = ImagesLoss(train_features[1], train_labels[1], blur=1e-3,scaling=0.9)
            # Loss = SamplesLoss("sinkhorn", p=2, blur = 1e-3, scaling = 0.9, debias=True)
            # loss = Loss(train_features[1], train_labels[1])
            # print('Shape of loss',loss.shape)
            Total_loss.append(loss)
            firstIndex.append(train_features[0])
            secondIndex.append(train_labels[0])
        wasserstein_distances = torch.hstack(Total_loss)
        wasserstein_firstIndex  = torch.hstack(firstIndex)
        wasserstein_secondIndex = torch.hstack(secondIndex)
        Id_max = torch.argmax(wasserstein_distances)
        return wasserstein_firstIndex[Id_max].item(),wasserstein_secondIndex[Id_max].item(),wasserstein_distances[Id_max].item()


    def save_fit(self, rootdir):
        """Save fit to load afterwards
        """
        import pickle
        pickle.dump(self.interpolator_tools, open(rootdir+'barycentric_interpolator_tools', "wb" ))
        np.save(rootdir+'fit_conv', self.fit_conv, allow_pickle=True)
        np.save(rootdir+'index_fit_fields', self.index_fit_fields, allow_pickle=True)
        for i, u in enumerate(self.fit_output):
            u.save(check_create_dir(rootdir+'u_'+str(i)+'/'))
        # Save uuid of training set
        with open(rootdir+"uuid_training.txt", "w") as uuid_training: # save unique identifier
            uuid_training.write(str(self.uuid_training))
        torch.save(self.final_barycenter, rootdir+'final_barycenter')

    def load_fit(self, rootdir):
        """Load fit to be able to call the predict method
        """
        import os
        import pickle
        # self.interpolator_tools = pickle.load(open(rootdir+'barycentric_interpolator_tools', "rb" ))
        self.fit_conv = np.load(rootdir+'fit_conv.npy', allow_pickle=True)
        self.index_fit_fields = np.load(rootdir+'index_fit_fields.npy', allow_pickle=True)
        self.fit_output = []
        subdirs = [x[0] for x in os.walk(rootdir) if '/u_' in x[0]]
        #print('subdirs',subdirs)
        for subdir in natsorted(subdirs):
            #print('subdir',subdir)
            self.fit_output.append(DataStruct.load(subdir+'/'))
        # Load uuid of training set
        with open(rootdir+"uuid_training.txt", "r") as uuid_training: # save unique identifier
            self.uuid_training = uuid.UUID(uuid_training.read())
        # modify KNN
        for id in range(len(self.interpolator_tools)):
            #self.interpolator_tools[id]['parameters'] = np.array([param.tolist() for param in query.parameter])
            self.interpolator_tools[id]['nn']=NearestNeighbors(n_neighbors=5, algorithm='ball_tree').fit(self.interpolator_tools[id]['parameters'])
        torch.load(rootdir+'final_barycenter', map_location=device)

    
    def IWD_interpolation(self,X,Y,X_new,p=2):
        X_new = X_new.reshape(1,-1)
        alpha = np.linalg.norm(X-X_new,axis=1)
        min_val = np.min(alpha)
        idx = np.argmin(alpha)
        if min_val <= 1.e-5:
            print('close point')
            alpha = np.zeros(X.shape[0])
            alpha[idx] = 1
        else:
            alpha = 1./alpha**p
            alpha = alpha/np.sum(alpha)
        # construct Y_new
        Y_new = alpha.dot(Y)
        return Y_new


    def core_predict(self, q: QueryDataStruct, n=None, args_rbf={'function': 'multiquadric', 'smooth': 0.}):
        """
        @param query: QueryDataStruct with information about
            * parameters: paramter values
            * field_spatial_coordinates: coordinates where we want to infer field values. (n_coords_infer, spatial_dim)
        @return: predicted field at field_spatial_coordinates
        """
        D = self.fit_output[0].field[0].dim()
        interpolator_tools = None
        if n is None:
            n = len(self.interpolator_tools)-1
            interpolator_tools = self.interpolator_tools[-1]
        elif n>len(self.interpolator_tools)-1:
            n = len(self.interpolator_tools)-1
            interpolator_tools = self.interpolator_tools[-1]
        else:
            interpolator_tools = self.interpolator_tools[n]

        # Find nearest q's neighbors
        distances, index_neighbors = interpolator_tools['nn'].kneighbors([q.parameter[0].tolist()])
        if distances.flatten()[0] == 0.:
            print('Query belongs to training set. Everything was already precomputed.')
            # If query belongs to training set, we already have its weights.
            idx = index_neighbors.flatten()[0]
            weights = torch.tensor(interpolator_tools['weights'][idx], dtype=dtype, device=device)
            weights = weights[None,:]
        else:
            # If query not in training set, we infer the weights by local interpolation
            neighbors = interpolator_tools['parameters'][index_neighbors.flatten(), :]
            values = interpolator_tools['weights'][index_neighbors.flatten(), :]
            #DEBUG:
            # print('Values in core_predict',values)
            # Rbf interpolation
            n_weights = interpolator_tools['weights'].shape[1]
            #DEBUG:
            # print('interpolator_tools_weights',interpolator_tools['weights'].shape)
            weights = torch.empty(n_weights, dtype=dtype, device=device)
            for i in range(n_weights):
                XY = [ neighbors[:,j] for j in range(neighbors.shape[1])] + [values[:,i]]
                rbfi = scipy.interpolate.Rbf(*XY, **args_rbf) # The input format here must be scipy.interpolate.Rbf(x, y, z, values)
                ci = rbfi(*q.parameter[0].tolist()) # The input format must be a list of parameter coefficients which we must unpack: rbfi(param_c[0], ...,param_c[k])
                weights[i]= ci.flatten()[0]
            #weights[-1] = 1.-torch.sum(weights[:-1])
            # projection
            weights=weights[None,:]
            weights.data= mat2simplex(weights)
            #print('w',torch.sum(weights,weights))

            # IWD interpolation
            # w = self.IWD_interpolation(neighbors,values,q.parameter[0].cpu().numpy())
            # weights = torch.tensor(w, device=device)
            #print('w',w)
        #bary = ImagesBarycenterWithGrad(fields=[u.field[0] for u in self.fit_output[:n+2]],weights= weights)
        #DEBUG:

        if D == 1:
            measures= torch.cat([u.field[0][None,None,:] for u in self.fit_output[:n+2] ], dim =1)
            bary  = ImagesBarycenter_1d(measures=measures, weights=weights,**self.params_sinkhorn_bary)
            #DEBUG:
            # print('Data type of bary in core_predict',type(bary))
            # print('Shape of bary in core_predict',bary.shape)
            return bary.squeeze()
        else:
            measures= torch.cat([u.field[0][None,None,:,:] for u in self.fit_output[:n+2] ], dim =1)
            bary  = ImagesBarycenter_v2(measures=measures, weights=weights,**self.params_sinkhorn_bary)
            return bary[0,0]


    def weights_core_predict(self, q: QueryDataStruct,target:TargetDataStruct, n=None):
        """
        @param query: QueryDataStruct with information about
            * parameters: paramter values
            * field_spatial_coordinates: coordinates where we want to infer field values. (n_coords_infer, spatial_dim)
        @return: predicted field at and weights
        """
        interpolator_tools = None
        if n is None:
            n = len(self.interpolator_tools)-1
            interpolator_tools = self.interpolator_tools[-1]
        elif n>len(self.interpolator_tools)-1:
            n = len(self.interpolator_tools)-1
            interpolator_tools = self.interpolator_tools[-1]
        else:
            interpolator_tools = self.interpolator_tools[n]

        
        D = self.fit_output[0].field[0].dim()
        
        if D == 1:
            measures= torch.cat([u.field[0][None,None,:] for u in self.fit_output[:n+2] ], dim =1)
            barycenter, weight, evolution = projGraFixSupp(
                            target_field=target.field[0][None,None,:],
                            measures = measures,
                            field_coordinates=q.field_spatial_coordinates[0])
            
                            # params_opt = self.params_opt_best_barycenter,
                            # params_sinkhorn_bary=self.params_sinkhorn_bary)
            
            result = {'barycenter':barycenter.squeeze(), 'weight':weight[0], 'loss':evolution['loss'][-1]}
            
        else:
            measures= torch.cat([u.field[0][None,None,:,:] for u in self.fit_output[:n+2] ], dim =1)
            barycenter, weight, evolution = projGraFixSupp(
                            target_field=target.field[0][None,None,:,:],
                            measures = measures,
                            field_coordinates=q.field_spatial_coordinates[0],
                            params_opt = self.params_opt_best_barycenter,
                            params_sinkhorn_bary=self.params_sinkhorn_bary)
        
            result = {'barycenter':barycenter[0,0], 'weight':weight[0], 'loss':evolution['loss'][-1]}
        
        return result
