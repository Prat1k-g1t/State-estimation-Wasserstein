# SPDX-FileCopyrightText: The sinkhorn-rom authors, see file AUTHORS
#
# SPDX-License-Identifier: GPL-3.0-or-later

import pykeops
pykeops.clean_pykeops()
print(pykeops.__version__)            # Should be 1.5
pykeops.test_numpy_bindings()   # Should tell you sth like “bindings OK"
pykeops.test_torch_bindings()     # Should tell you sth like “bindings OK"



#from geomloss import SamplesLoss
import argparse

from ..lib.Models.Embedding import Embedding
from ..lib.Benchmarks.Benchmark import *
from ..lib.DataManipulators.Problems import *


# List of available problems
problem_dict = {'Gaussian1d': Gaussian1d,
                'Gaussian2d': Gaussian2d,
                'VlasovPoisson': VlasovPoisson,
                'Transport1d': Transport1d,
                'Burger2d'   : Burger2d,
                'Burger1d'   : Burger1d,
                }

# Parse
parser = argparse.ArgumentParser()
parser.add_argument('-p', type=str, default='Burger2d', help=', '.join(problem_dict.keys()))
parser.add_argument('-nfit',nargs='+', type=int, default=100, help='number of parameters fit')
parser.add_argument('-idfit', type=int, default=0, help='id set fit')
parser.add_argument('-np', type=int, default=100, help='number of parameters predict')
parser.add_argument('-idp', type=int, default=1, help='id set predict')
args = parser.parse_args()

# Params for sinkhorn

#Loss = SamplesLoss("sinkhorn", blur=5.e-3, scaling=.9)
params_sinkhorn_bary = {'blur': 0.001, 'p':2, 'scaling_N':100,'backward_iterations':5} 
params_opt_best_barycenter = {'optimizer': 'Adam','lr': 0.02,'nmax': 200, 'type_loss':'mse','gamma':1,'k_sparse':10}
params_online = {'n_neighbors':10, 'method':'AS'}
params_embedding = {'nits': 1,'k':98, 'robust':True, 'cvx':True}

#params_regression    = {'type_regression':'Shepard', 'reg_param': 5.e-3,'n_neighbors':20, 'adaptive_knn':False,'power':4}
#params_kernel       = {'metric':'rbf', 'gamma':10}

# Problem and models to consider
# We put them on a list since Benchmark iterates over them


#Debug
print('args.nfit', args.nfit)
problems = [Problem(name=args.p,
                    id='Predict_weights_LE_train100_test100_knn10_optim',
                    config_set_fit={'nparam': args.nfit, 'id_set': args.idfit},
                    config_set_predict={'nparam': args.np, 'id_set': args.idp})
            ]

models = [Embedding(params_embedding=params_embedding,params_online = params_online, params_sinkhorn_bary=params_sinkhorn_bary, params_opt_best_barycenter =params_opt_best_barycenter),]

for model in models:
    for problem in problems:
        fit(problem, model)
        predict_weights(problem, model)
        #sample_visualization(problem,model)
        #analysis_weights_knn(problem, model)
    #plots_predict_images(problems, model)

