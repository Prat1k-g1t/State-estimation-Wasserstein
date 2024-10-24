# SPDX-FileCopyrightText: The sinkhorn-rom authors, see file AUTHORS
#
# SPDX-License-Identifier: GPL-3.0-or-later

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_spd_matrix
import ot
from ..lib.Evaluators.Optimal_transport import  semi_discrete_ot
from ..config import results_dir
from ..lib.DataManipulators.Problems import *
import argparse
from geomloss import SamplesLoss
from ..lib.Evaluators.Barycenter import ImagesLoss

def get_embedding_distances(point_clouds,masses=None, grid_size=400):
    """
    Returns the pairwise distances of 'point clouds' based on
    embedding distances.
    Args:
        point_clouds (dict): dictionary of point clouds
        grid_size (int): discretization parameter for the
                         transport plans (m in the paper)
    Returns:
        distances (array): list of pairwise embedding ditances
    """
    nb_clouds = len(point_clouds)
    # compute transport maps
    sd_ot = semi_discrete_ot(grid_size=grid_size)
    sd_ot.fit_transport_plans(point_clouds,masses)
    # deduce transport map distances
    distances = []
    for i in range(nb_clouds-1):
        for j in range(i+1, nb_clouds):
            dist_2 = np.sum( (sd_ot.transport_plans[i]\
                            - sd_ot.transport_plans[j])**2 )
            distances.append( np.sqrt((1/(grid_size**2)*dist_2)) )
    return np.array(distances)

def get_W2_distances(point_clouds, masses = None, verbose=True):
    """
    Returns the pairwise distances of 'point clouds' based on
    W2 distances (computed with the POT library).
    Args:
        point_clouds (dict): dictionary of point clouds
        verbose (boal): wether or not print computation advancment
    Returns:
        distances (array): list of pairwise Wasserstein ditances
    """
    nb_clouds = len(point_clouds)
    tot = nb_clouds * (nb_clouds-1) / 2
    count = 0
    # Define uniform weights for source and target measures
    #nb_points = point_clouds[0].shape[0]
    # a = np.ones(nb_points)/nb_points
    # b = a.copy()
    # Wasserstein distances
    wasserstein_distances = []
    for i in range(nb_clouds-1):
        x = point_clouds[i]
        nb_points = point_clouds[i].shape[0]
        if masses is None:
            a = np.ones(nb_points)/nb_points
        else:
            a = masses[i]

        for j in range(i+1, nb_clouds):
            nb_points = point_clouds[j].shape[0]
            if masses is None:
                b = np.ones(nb_points)/nb_points
            else:
                b = masses[j]
            count += 1
            y = point_clouds[j]
            C = (x[:, 0].reshape(-1,1) - y[:, 0].reshape(1,-1))**2 \
                + (x[:, 1].reshape(-1,1) - y[:, 1].reshape(1,-1))**2
            # W2 computed with POT
            wasserstein_distances.append( np.sqrt(ot.emd2(a, b, C)) )
            if verbose:
                print("Computed {:3.1f}%".format(100*count/tot))
    return np.array(wasserstein_distances)


def geomloss_get_W2_distances(point_clouds, masses = None, verbose=True):
    """
    Returns the pairwise distances of 'point clouds' based on
    W2 distances (computed with the POT library).
    Args:
        point_clouds (dict): dictionary of point clouds
        verbose (boal): wether or not print computation advancment
    Returns:
        distances (array): list of pairwise Wasserstein ditances
    """
    nb_clouds = len(point_clouds)
    tot = nb_clouds * (nb_clouds-1) / 2
    count = 0
    # Wasserstein distances
    eps = 5.e-3
    Loss = SamplesLoss("sinkhorn", blur=eps, scaling=.9)
    wasserstein_distances = []
    for i in range(nb_clouds-1):
        x = point_clouds[i]
        nb_points = point_clouds[i].shape[0]
        if masses is None:
            a = np.ones(nb_points)/nb_points
        else:
            a = masses[i]

        for j in range(i+1, nb_clouds):
            y = point_clouds[j]
            nb_points = point_clouds[j].shape[0]
            if masses is None:
                b = np.ones(nb_points)/nb_points
            else:
                b = masses[j]
            count += 1
            
            # C = (x[:, 0].reshape(-1,1) - y[:, 0].reshape(1,-1))**2 \
            #     + (x[:, 1].reshape(-1,1) - y[:, 1].reshape(1,-1))**2
            # W2 computed with 
            print('a,b',a.shape,b.shape)
            wasserstein_distances.append( np.sqrt(Loss(a.flatten(),x, b.flatten(), y).item()) )
            if verbose:
                print("Computed {:3.1f}%".format(100*count/tot))
    return np.array(wasserstein_distances)
def imageloss_get_W2_distances(point_clouds, masses , verbose=True):
    """
    Returns the pairwise distances of 'point clouds' based on
    W2 distances (computed with the POT library).
    Args:
        point_clouds (dict): dictionary of point clouds
        verbose (boal): wether or not print computation advancment
    Returns:
        distances (array): list of pairwise Wasserstein ditances
    """
    nb_clouds = len(point_clouds)
    tot = nb_clouds * (nb_clouds-1) / 2
    count = 0
    # Wasserstein distances
    eps = 5.e-3
    Loss = SamplesLoss("sinkhorn", blur=eps, scaling=.9)
    wasserstein_distances = []
    for i in range(nb_clouds-1):
        a = masses[i]
        for j in range(i+1, nb_clouds):
            b = masses[j]
            count += 1
            # W2 computed with 
            wasserstein_distances.append( np.sqrt(ImagesLoss(a[None,None,:,:],b[None,None,:,:],blur=0.001,scaling=0.8).item()) )
            if verbose:
                print("Computed {:3.1f}%".format(100*count/tot))
    return np.array(wasserstein_distances)

def plot_distances_comparison(wasserstein_distances, embedding_distances, title, pdf_title):
    """
    Plot distances comparison.
    Args:
        wasserstein_distances (array): list of pairwise Wasserstein distances
        embedding_distances (array): list of pairwise embedding distances
        title (str): figure title
        pdf_title (str): title for save file
    """
    f = plt.figure(figsize=(12, 8))
    plt.rcParams.update({'font.size': 25})
    # scatter distances
    plt.scatter(wasserstein_distances, embedding_distances, s=3.0)
    # add 'y=x'
    _min = min(np.min(wasserstein_distances), np.min(embedding_distances))
    _max = max(np.max(wasserstein_distances), np.max(embedding_distances))
    plt.plot([0.9*_min, 1.1*_max], [0.9*_min, 1.1*_max], '--', label='y = x', linewidth=0.8, c="r")
    # legend
    plt.xlabel(r"$W_2(\mu_M, \nu_N)$")
    plt.ylabel(r"$||T_{\mu_M} - T_{\nu_N}||_{L^2(\rho)}$") 
    plt.title(title)
    plt.legend()
    plt.show()
    # save fig
    f.savefig(results_dir+pdf_title, bbox_inches='tight')

## Gaussian test case
# nb_clouds = 10 
# nb_points = 300
# gaussian_point_clouds = {}
# sigma = 1e-2

# np.random.seed(2)

# for c in range(nb_clouds):
#     mean = 0.25 + 0.5*np.random.rand(2) 
#     cov = sigma * make_spd_matrix(2)
#     gaussian_point_clouds[c] = np.random.multivariate_normal(mean, cov, nb_points)
# print(gaussian_point_clouds)

# gaussian_embedding_distances = get_embedding_distances(gaussian_point_clouds)
# gaussian_wasserstein_distances = get_W2_distances(gaussian_point_clouds, verbose=False)
# plot_distances_comparison(gaussian_wasserstein_distances, gaussian_embedding_distances,title="Gaussians", pdf_title="W2_dT_gaussians.pdf")

## Burger test case
# List of available problems
problem_dict = {'Gaussian1d': Gaussian1d,
                'Gaussian2d': Gaussian2d,
                'VlasovPoisson': VlasovPoisson,
                'Transport1d': Transport1d,
                'Burger2d'   : Burger2d
                }


parser = argparse.ArgumentParser()
parser.add_argument('-p', type=str, default='Gaussian2d', help=', '.join(problem_dict.keys()))
parser.add_argument('-nfit',  type=int, default=10, help='number of parameters fit')
parser.add_argument('-idfit', type=int, default=0, help='id set fit')
parser.add_argument('-np', type=int, default=10, help='number of parameters predict')
parser.add_argument('-idp', type=int, default=1, help='id set predict')
args = parser.parse_args()

problem = Problem(name=args.p,
        id='distance_example',
        config_set_fit={'nparam': args.nfit, 'id_set': args.idfit},
        config_set_predict={'nparam': args.np, 'id_set': args.idp})

field_coordinates, snapshots, parameters, uuid = problem.load_dataset(**problem.config_set_fit)
print('X',field_coordinates.shape)
print('params',parameters)
print('fields',snapshots[0].shape)

nb_clouds = len(snapshots)
Burger_point_clouds = {}
Burger_masses = {}
Pytorch_Burger_point_clouds = {}
Pytorch_Burger_masses = {}

for c in range(nb_clouds):
    Burger_masses[c]=snapshots[c].flatten().detach().cpu().numpy()
    Burger_point_clouds[c]=field_coordinates.detach().cpu().numpy()
    Pytorch_Burger_masses[c]=snapshots[c]
    Pytorch_Burger_point_clouds[c]=field_coordinates
#print(Burger_point_clouds)

Burger_embedding_distances = get_embedding_distances(Burger_point_clouds, Burger_masses)
print(Burger_embedding_distances)
Burger_wasserstein_distances = geomloss_get_W2_distances(Pytorch_Burger_point_clouds,Pytorch_Burger_masses, verbose=False)
print(Burger_wasserstein_distances)
plot_distances_comparison(Burger_wasserstein_distances, Burger_embedding_distances,title="Gaussian40", pdf_title="W2_dT_Gaussian40_2.pdf")




