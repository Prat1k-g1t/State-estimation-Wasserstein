# SPDX-FileCopyrightText: The sinkhorn-rom authors, see file AUTHORS
#
# SPDX-License-Identifier: GPL-3.0-or-later

import pykeops
pykeops.clean_pykeops()
print(pykeops.__version__)            # Should be 1.5
pykeops.test_numpy_bindings()   # Should tell you sth like “bindings OK"
pykeops.test_torch_bindings()     # Should tell you sth like “bindings OK"



from geomloss import SamplesLoss
import argparse

from ..lib.Models.NonIntrusiveGreedyImages import NonIntrusiveGreedyImages 

#from ..lib.Models.WeightGreedyImages import WeightGreedyImages 
from ..lib.DataManipulators.Problems import Problem
from ..lib.Benchmarks.Benchmark import *
from ..lib.DataManipulators.Problems import *

# List of available problems
problem_dict = {'Gaussian1d'      : Gaussian1d,
                'Gaussian2d'      : Gaussian2d,
                'VlasovPoisson'   : VlasovPoisson,
                'Burger2d'        : Burger2d,
                'Burger1d'        : Burger1d,
                'KdV1d'           : KdV1d,
                'ViscousBurger1d' : ViscousBurger1d,
                'CamassaHolm1d'   : CamassaHolm1d
                }

# Parse
parser = argparse.ArgumentParser()
parser.add_argument('-p', type=str, default='Burger2d', help=', '.join(problem_dict.keys()))
parser.add_argument('-nfit', type=int, default=100, help='number of parameters fit')
parser.add_argument('-idfit', type=int, default=0, help='id set fit')
parser.add_argument('-np', type=int, default=100, help='number of parameters predict')
parser.add_argument('-idp', type=int, default=1, help='id set predict')
args = parser.parse_args()

# Params for sinkhorn
eps = 1.e-3
Loss = SamplesLoss("sinkhorn", blur = eps, scaling=.9, debias=True) #, blur = eps
# Loss = SamplesLoss("sinkhorn", p=2, blur=0.001, debias=True)
params_sinkhorn_bary = {'blur': 0.0, 'p':2, 'scaling_N':100,'backward_iterations':5} 
params_opt_best_barycenter = { 'optimizer': 'Adam','lr': 0.01,'nmax': 50,'type_loss':'W2','gamma':1,'k_sparse':5}

# Problem and models to consider
# We put them on a list since Benchmark iterates over them
#myid = 'Greedy'+'_nfit' + str(args.nfit) + '_np'+str(args.np) 

problems = [ Problem(name=args.p,
                    # id='Greedytest_burger2d_M64_ns100_nmax9_again',
                    id='Greedytest_'+str(args.p)+'_N'+str(args.nfit),
                    config_set_fit={'nparam': args.nfit, 'id_set': args.idfit},
                    config_set_predict={'nparam': args.np, 'id_set': args.idp}),
]
models = [
    NonIntrusiveGreedyImages ( Loss = Loss, nmax = 10,
                    compute_intermediate_interpolators = True,
                    params_sinkhorn_bary=params_sinkhorn_bary,
                    params_opt_best_barycenter=params_opt_best_barycenter),
]

for problem in problems:
    for model in models:
        # fit(problem, model)
        # predict(problem, model)
        pass
    plots_fit(problem, models)
    # plots_predict(problem, models)

