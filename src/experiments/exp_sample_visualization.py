# SPDX-FileCopyrightText: The sinkhorn-rom authors, see file AUTHORS
#
# SPDX-License-Identifier: GPL-3.0-or-later

import argparse
import torch

from ..lib.Benchmarks.Benchmark import *
from ..lib.DataManipulators.Problems import *
from ..lib.Evaluators.Barycenter import  ImagesLoss, ImagesBarycenter_v2, projGraFixSupp, projGraAdapSupp, projRGraSP, mat2simplex
from ..config import device, dtype
from ..visualization import plot_sample_images

import numpy as np
from tqdm import tqdm

print('check cuda', torch.cuda.is_available())


# List of available problems
problem_dict = {'Gaussian1d': Gaussian1d,
                'Gaussian2d': Gaussian2d,
                'VlasovPoisson': VlasovPoisson,
                'Transport1d': Transport1d,
                'Burger2d'   : Burger2d
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


params_sinkhorn_bary = {'blur': 0.001, 'p':2, 'scaling_N':100,'backward_iterations':5} 
params_opt_best_barycenter = {'optimizer': 'Adam','lr': 0.02,'nmax': 300, 'type_loss':'W2','gamma':1,'k_sparse':10}

#params_KNN = {'n_neighbors':100}


problem = Problem(name=args.p, id='Sample_visualization_train100',
        config_set_fit={'nparam': args.nfit, 'id_set': args.idfit},
        config_set_predict={'nparam': args.np, 'id_set': args.idp})

root_dir = check_create_dir(results_dir+'{}/{}/'.format(problem.name, problem.id))

field_coordinates_train, snapshots_train, parameters_train, _ = problem.load_dataset(**problem.config_set_fit)
query_train = QueryDataStruct(parameter=parameters_train,field_spatial_coordinates=[field_coordinates_train])
target_train = TargetDataStruct(field=snapshots_train)

# field_coordinates_test, snapshots_test, parameters_test, _ = problem.load_dataset(**problem.config_set_predict)
# query_test = QueryDataStruct(parameter=parameters_test,field_spatial_coordinates=[field_coordinates_test])
# target_test = TargetDataStruct(field=snapshots_test)

# points_train = torch.Tensor(parameters_train.detach().cpu().numpy())
# points_test = torch.Tensor(parameters_test.detach().cpu().numpy())
# print(parameters_test[0:10])
# print('select_parameter',parameters_test[8])
# print(points_test[8])

n_tests = 100
num_predicts = 9
sample_ids = random.sample(np.arange(n_tests).tolist(), num_predicts)
indexes = np.array(sample_ids)
sample_fields = [snapshots_train[id].detach().cpu().numpy() for id in sample_ids]
print('indexes',indexes)
print('samples',sample_fields)
fig_opts={'colors': [], 'titles': None, 'rootdir': root_dir,  'fig_title': 'sample_fields', 'format': '.pdf'}
plot_sample_images(indexes=indexes,fields=sample_fields,fig_opts=fig_opts)