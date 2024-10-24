# SPDX-FileCopyrightText: The sinkhorn-rom authors, see file AUTHORS
#
# SPDX-License-Identifier: GPL-3.0-or-later

import torch
import numpy as np
from scipy.spatial.distance import cdist
from itertools import product
import time
from scipy.sparse import csr_matrix

class PointCloud:

    @staticmethod
    def split_point_cloud(points, P=2, spatial_metric=2):
        """
            Partition of the point cloud into PxP uniform cells
        """
        coord_bounds, lengths_cell, dx = [], [], []
        dim = points.shape[1]
        for d in range(dim):
            coord_bounds.append( [torch.min(points[:,d]).item(), torch.max(points[:,d]).item()] )
            lengths_cell.append( coord_bounds[-1][1]-coord_bounds[-1][0] + 1.e-9)
            dx.append( lengths_cell[-1]/P ) # length of each cell
        radius = torch.norm(torch.tensor([h**2 for h in dx]), p=spatial_metric).item()

        # Later on, we will need to compute Kij = exp(-||xi-xj||^2/sigma^2) for all points.
        # This is a very heavy operation because we have 79053 points
        # As soon as ||xi-xj||^2/sigma^2>=20, Kij \approx 1.e-9 so we can round it to zero
        # and skip the operation. This motivates to introduce a uniform coarse grid of PxP cells
        # where we iterate only between the points inside each cell and the neighboring ones.

        # Create cells
        cells = []
        for idx_cell in product(np.arange(P), repeat=dim):
            cells.append({'idx': idx_cell,
                          'xmin': [ bound_x[0]+idx_cell[idx_coord]*dx[idx_coord] for idx_coord, bound_x in enumerate(coord_bounds) ],
                          'xmax': [ bound_x[0]+(1+idx_cell[idx_coord])*dx[idx_coord] for idx_coord, bound_x in enumerate(coord_bounds) ],
                          'dx': dx,
                          'spatial_metric': spatial_metric,
                          'radius': radius,
                          'points':[],
                          'idx_points':[]
                        })

        # Add points to cells
        for i, p in enumerate(points):
            K = []
            for d in range(dim):
                K.append( int(((p[d] - coord_bounds[d][0])//dx[d]).item()) )
            c = next(c for c in cells if list(c["idx"]) == K)
            c['points'].append(p)
            c['idx_points'].append(i)

        return cells

    @staticmethod
    def find_cell_k_neighbors(cells, k=1):
        """Find k-nearest neighbors of each cell in cells
        """
        delta_idx = np.arange(-k, k+1)
        for i, ci in enumerate(cells):
            ci['neighbors'] = []
            for j, cj in enumerate(cells):
                if cj['points']:
                    d = tuple(np.array(ci['idx'])-np.array(cj['idx']))
                    if d in product(delta_idx, repeat=len(cj['points'][0])):
                        ci['neighbors'].append(cj)
        return cells

    @staticmethod
    def compute_sparse_C(cells, threshold = 1.e-14, eps=1.e-2, spatial_metric=2):
        """Given a threshold, find neighboring cells upon which we will have to iterate.
        """
        # Kernel function
        kf = lambda x, y: np.exp(-np.linalg.norm(x-y)**2/eps)
        kfd = lambda d: np.exp(-np.linalg.norm(d)**2/eps)
        invkfd = lambda y: eps*np.sqrt(-np.log(y))

        tic = time.time()
        dmin = invkfd(threshold)
        kn = int(1 + dmin // min( cells[0]['dx']) ) # depth of neighboord. Eg: kn=1-> take immediate neighboring cells.
        cells = PointCloud.find_cell_k_neighbors(cells, k=kn)
        toc = time.time()

        # Fill in Ksp (sparse matrix version)
        tic = time.time()
        #Kmat = np.zeros((field_coordinates.shape[0], field_coordinates.shape[0])) # Full matrix (comment if you work with sparse version)
        Ksp_entries = []
        Csp_entries = []
        rows = []
        columns = []
        for ic, c in enumerate(cells):
            if c['points']:
                for c_ngb in c['neighbors']:
                    for i, p in enumerate(c['points']):
                        idx_p = c['idx_points'][i]
                        for j, p_ngb in enumerate(c_ngb['points']):
                            kval = kf(p, p_ngb)
                            if kval >= threshold:
                                idx_p_ngb = c_ngb['idx_points'][j]
                                #Kmat[idx_p, idx_p_ngb] = kf(p, p_ngb) # Full matrix (comment if you work with sparse version)
                                Ksp_entries.append(kf(p, p_ngb))
                                # print(torch.tensor(p))
                                Csp_entries.append(torch.norm(p-p_ngb, p=2)**2)
                                rows.append(idx_p)
                                columns.append(idx_p_ngb)
        n_points = sum([ len(c['points']) for c in cells ])
        # Ksp = csr_matrix((Ksp_entries, (rows, columns)), shape=(n_points, n_points)) # Sparse matrix
        # Csp = csr_matrix((Csp_entries, (rows, columns)), shape=(n_points, n_points)) # Sparse matrix

        indices = torch.LongTensor([rows, columns])
        values = torch.FloatTensor(Csp_entries)
        Csp_torch = torch.sparse_coo_tensor(indices, values, torch.Size([n_points, n_points]))

        values = torch.FloatTensor(Ksp_entries)
        Ksp_torch = torch.sparse_coo_tensor(indices, values, torch.Size([n_points, n_points]))
        toc = time.time()
        print('Computed Csp in {} s.'.format(toc-tic))
        # print(Csp_torch)
        return Ksp_torch, Csp_torch