# SPDX-FileCopyrightText: The sinkhorn-rom authors, see file AUTHORS
#
# SPDX-License-Identifier: GPL-3.0-or-later

import time
import torch
from typing import List
from ...config import use_pykeops, device, dtype, use_cuda

if use_pykeops:
    from pykeops.torch import Genred

    # Logsumexp with PyKeops
    def logsumexp_template(nx, ny):
        formula = '(-SqNorm2(x-y)+fi+gj)/eps'
        variables = ['x = Vi({})'.format(nx),  # First arg   : i-variable, of size d (scalar)
                    'y = Vj({})'.format(ny),  # Second arg  : j-variable, of size d (scalar)
                    'fi = Vi(1)',    # Third arg   : i-variable, of size 1 (scalar)
                    'gj = Vj(1)',    # Fourth arg   : j-variable, of size 1 (scalar)
                    'eps = Pm(1)']  # Fifth  arg : Parameter,  of size 1 (scalar)
        return [Genred(formula, variables, reduction_op='LogSumExp', axis=0, dtype = "float64"),
                Genred(formula, variables, reduction_op='LogSumExp', axis=1, dtype = "float64")]


def barycenter_pykeops(fields: List[torch.Tensor],
                        weights: torch.Tensor,
                        field_coordinates: torch.Tensor,
                        logsumexp= None,
                        spatial_metric=2,
                        nmax=100,
                        eps=1.e-2,
                        tau1= 10,
                        tau2=10,
                        C: torch.Tensor = None):
    """
        Compute barycenter
    """
    if spatial_metric != 2:
        raise Exception("Computation of barycenter with barycenter_pykeops only supported for spatial_metric=2 and current value is {}".format(spatial_metric))

    tic = time.time()

    if logsumexp is None:
        print('Computing logsumexp expression for pykeops inside barycenter_pykeops. This may result in a loss of computing time.')
        logsumexp = logsumexp_template(field_coordinates.shape[1], field_coordinates.shape[1])

    if use_cuda:
        backend = 'GPU'
    else:
        backend = 'CPU'

    # Initializations
    nfields = len(fields)
    dofs = fields[0].shape[0]
    t1 = tau1/(tau1+eps)
    t2 = tau2/(tau2+eps)

    Fl = [torch.zeros(dofs, dtype=dtype, device=device)]*nfields
    Gl = [torch.zeros(dofs, dtype=dtype, device=device)]*nfields
    softmins = [torch.zeros(dofs, dtype=dtype, device=device)]*nfields
    mlogeps_bary = torch.zeros(dofs, dtype=dtype, device=device)

    # Sinkhorn iterations
    for i in range(nmax):
        for j, field in enumerate(fields):
            Fl[j] = Fl[j] + t1*(eps*torch.log(field) -eps*logsumexp[1](field_coordinates,
                                                                    field_coordinates,
                                                                    Fl[j].unsqueeze(1)/t1,
                                                                    Gl[j].unsqueeze(1),
                                                                    torch.tensor([eps], dtype=dtype, device=device),
                                                                    backend=backend).squeeze())
            softmins[j] = -eps*logsumexp[0](field_coordinates,
                                        field_coordinates,
                                        Fl[j].unsqueeze(1),
                                        -mlogeps_bary.unsqueeze(1),
                                        torch.tensor([eps], dtype=dtype, device=device),
                                        backend=backend).squeeze()
        
        mlogeps_bary = mlogeps_bary - torch.matmul(torch.stack(softmins, dim=1), weights)
        
        for j, gl in enumerate(Gl):
            Gl[j] = Gl[j] + t2*(mlogeps_bary -eps*logsumexp[0](field_coordinates,
                                                            field_coordinates,
                                                            Fl[j].unsqueeze(1),
                                                            Gl[j].unsqueeze(1)/t2,
                                                            torch.tensor([eps], dtype=dtype, device=device),
                                                            backend=backend).squeeze())

    toc = time.time()
    return torch.exp(mlogeps_bary/eps)

def barycenter_fallback(fields: List[torch.Tensor],
                        weights: torch.Tensor,
                        field_coordinates: torch.Tensor,
                        logsumexp= None,
                        spatial_metric=2,
                        nmax=100,
                        eps=1.e-2,
                        tau1= 10,
                        tau2=10,
                        C: torch.Tensor = None):

    # Auxiliary funcitons
    def softmin(A, eps, axis=1):
        """
            axis=1: sum in logsumexp is taken along columns --> gives softmin value for each of the rows
        """
        return -eps*torch.logsumexp(-A/eps, axis=axis)
    
    def sinkhorn_diff(C, F, G):
        return C[:,:,None]-F[:,None,:]-G[None,:,:]

    def sinkhorn_diff_elementwise(C, f, g):
        return C - torch.outer(f, torch.ones(f.shape)) - torch.outer(torch.ones(g.shape), g)
    
    tic = time.time()
    
    # Get Device and Type
    # dtype = fields[0].dtype
    # device = fields[0].device
    # dtype = dtype
    # device = device

    # Initializations
    nfields = len(fields)
    dofs = fields[0].shape[0]
    t1 = tau1/(tau1+eps)
    t2 = tau2/(tau2+eps)

    fields = torch.stack(fields, dim=1)
    Gl = torch.zeros(fields.shape, dtype=dtype, device=device)
    Fl = torch.zeros(fields.shape, dtype=dtype, device=device)
    ones = torch.ones(nfields, dtype=dtype, device=device)
    mlogeps_bary = torch.zeros(dofs, dtype=dtype, device=device)

    if C is None:
        C = torch.cdist(field_coordinates, field_coordinates, p=spatial_metric)**2

    # Sinkhorn iterations
    for i in range(nmax):
        Fl = Fl + t1*( eps*torch.log(fields) + softmin(sinkhorn_diff(C, Fl/t1, Gl), eps, axis=1)  )
        mlogeps_bary = mlogeps_bary - torch.matmul(softmin(sinkhorn_diff(C, Fl, -torch.ger(mlogeps_bary, ones)), eps, axis=0), weights)
        Gl = Gl + t2*(torch.ger(mlogeps_bary, ones) + softmin(sinkhorn_diff(C, Fl, Gl/t2), eps, axis=0))

    toc = time.time()
    return torch.exp(mlogeps_bary/eps)