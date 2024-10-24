# SPDX-FileCopyrightText: The sinkhorn-rom authors, see file AUTHORS
#
# SPDX-License-Identifier: GPL-3.0-or-later

import numpy as np
import torch
from scipy.spatial.distance import cdist

from .Sinkhorn import Sinkhorn

class Metrics:
    @staticmethod
    def mse(x, y):
        """

        @param x: vector 1
        @param y: vector 2
        @return:
        """

        return np.nanmean((x-y)**2)

    @staticmethod
    def dist(x, y, metric='euclidean'):
        return cdist(x, y, metric=metric)

    @staticmethod
    def wasserstein(fields: torch.Tensor,
                    field_coordinates: torch.Tensor,
                    C: torch.Tensor = None,
                    spatial_metric=2,
                    eps=1.e-2,
                    tau1= 10,
                    tau2=10,
                    nmax=100,
                    ):

        """
            Unbalanced Wasserstein loss
        """
        dtype = fields.dtype
        device = fields.device

        # Auxiliary funcitons
        def softmin(A, eps, axis=1):
            """
                axis=1: sum in logsumexp is taken along columns --> gives softmin value for each of the rows
            """
            return -eps*torch.logsumexp(-A/eps, axis=axis)
        
        def sinkhorn_diff(C, f, g):
            return C - torch.ger(f, torch.ones(f.shape, dtype=dtype, device=device)) - torch.ger(torch.ones(g.shape, dtype=dtype, device=device), g)
        
        # Initializations
        gl = torch.zeros(fields.shape[0], dtype=dtype, device=device)
        fl = torch.zeros(fields.shape[0], dtype=dtype, device=device)
        mcv: list = list() # Marginal constraint violation
        cost = torch.tensor(-1., dtype=dtype, device=device)
        t1 = tau1/(tau1+eps)
        t2 = tau2/(tau2+eps)
        if C is None:
            C = torch.cdist(field_coordinates, field_coordinates, p=spatial_metric)**2
        
        # Sinkhorn iterations
        for i in range(nmax):
            fl = fl + t1*(eps*torch.log(fields[:,0]) + softmin(sinkhorn_diff(C, fl/t1, gl), eps, axis=1))
            gl = gl+ t2*(eps*torch.log(fields[:,1]) + softmin(sinkhorn_diff(C, fl, gl/t2), eps, axis=0))
        P = torch.exp( -sinkhorn_diff(C, fl, gl)/eps )  # Transport plan P = np.diag(a)@ K @ np.diag(b)
        # cost = torch.sum(P*C) #+ tau1*KL(P@np.ones(a.shape), a) + tau2*KL(P.T@np.ones(b.shape), b)
        cost = torch.sum(P*C) + eps*torch.sum(P*(P-torch.ones(P.shape)))
        return cost, fl, gl, P, mcv