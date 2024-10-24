#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 14:57:36 2023

@author: prai
"""

# SPDX-FileCopyrightText: The sinkhorn-rom authors, see file AUTHORS
#
# SPDX-License-Identifier: GPL-3.0-or-later

from geomloss import SamplesLoss
# from sinkhorn_images import sinkhorn_divergence
import torch
import numpy as np
import argparse
import time
# from geomloss import SamplesLoss#ImagesBarycenter
from geomloss.sinkhorn_images import sinkhorn_divergence
from ..lib.DataManipulators.Problems import *
from ..lib.Evaluators.Barycenter import ImagesLoss, ImagesBarycenter_v2, \
    ImagesBarycenter_1d, projGraFixSupp, projGraAdapSupp
from ..lib.Models.NonIntrusiveGreedyImages import NonIntrusiveGreedyImages 
#from ..config import *
from ..config import results_dir, device, dtype
from ..visualization import plot_fields
from ..lib.Benchmarks.Benchmark import *

import matplotlib.pyplot as plt
plt.rcParams.update(plt.rcParamsDefault)
plt.rcParams["font.family"] = 'serif'
plt.rcParams["savefig.format"] = 'pdf'
plt.rcParams['text.usetex'] = True
import seaborn as sns
sns.set_theme(style="darkgrid")
# sns.set_style("darkgrid", {'axes.grid' : False})
sns.set(rc={"figure.dpi":100, 'savefig.dpi':100})
sns.set_context("paper")

# Used for visualising data and debugging
# def debug_visualization(x,U):
# #Used for debugging
#     fig = plt.figure()
#     for i in range(U.shape[1]):
#         plt.plot(x.numpy(),U[0][i][:].numpy())
#         plt.show()

problem_dict = {'Gaussian1d'      : Gaussian1d,
                'Gaussian2d'      : Gaussian2d,
                'Burger1d'        : Burger1d,
                'Burger2d'        : Burger2d,
                'KdV1d'           : KdV1d,
                'ViscousBurger1d' : ViscousBurger1d,
                'CamassaHolm1d'   : CamassaHolm1d
                }


# Parse
parser = argparse.ArgumentParser()
parser.add_argument('-p', type=str, default='Gaussian1d', help=', '.join(problem_dict.keys()))
parser.add_argument('-nfit', type=int, default=50, help='number of parameters fit')
parser.add_argument('-idfit', type=int, default=0, help='id set fit')
parser.add_argument('-np', type=int, default=50, help='number of parameters predict')
parser.add_argument('-idp', type=int, default=1, help='id set predict')
args = parser.parse_args()


# problem = [ Problem(name=args.p,
#                     id='Greedy_for_state_estimation_'+str(args.p),
#                     config_set_fit={'nparam': args.nfit, 'id_set': args.idfit},
#                     config_set_predict={'nparam': args.np, 'id_set': args.idp}),
# ]

problem = [ Problem(name=args.p,
                    # id='Greedytest_burger2d_M64_ns100_nmax9_again',
                    id='Greedytest_'+str(args.p)+'_N'+str(args.nfit),
                    config_set_fit={'nparam': args.nfit, 'id_set': args.idfit},
                    config_set_predict={'nparam': args.np, 'id_set': args.idp}),
]

loss = SamplesLoss("sinkhorn", blur = 1.e-3, p=2, scaling = 0.9, debias=True)

params_sinkhorn_bary = {'blur': 0.0, 'p':2, 'scaling_N':200,'backward_iterations':5} 
params_opt_best_barycenter = { 'optimizer': 'Adam','lr': 0.001,'nmax': 100,'type_loss':'W2','gamma':1,'k_sparse':5}

# Creating U_n using greedy algorithm
model = [
    NonIntrusiveGreedyImages ( Loss = loss, nmax = 10,
                    compute_intermediate_interpolators = False,
                    params_sinkhorn_bary=params_sinkhorn_bary,
                    params_opt_best_barycenter=params_opt_best_barycenter),
]

# fit(problem[0], model[0])
# state_estimation(problem[0], model)
load_greedy_measures(problem[0], model)
# state_estimation_travelling_soln(problem[0], model)
# plots_fit(problem[0], model)