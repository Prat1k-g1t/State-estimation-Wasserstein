#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 04:07:19 2023

@author: prai
"""
 
import numpy as np
import cbx as cbx
from cbx.dynamics import CBO
from cbx.objectives import Rastrigin
from cbx.utils.objective_handling import cbx_objective_fh
from cbx.scheduler import scheduler, multiply
# from cbx.plotting import plot_dynamic
import matplotlib.pyplot as plt

import torch

np.random.seed(420)
#%%
conf = {'alpha': 30.0,
        'dt': 0.01,
        'sigma': 8.1,#8,#5.1,#8.0,
        'lamda': 1.0,
        'batch_args':{
        'batch_size':200,
        'batch_partial': False},
        'd': 5,
        'max_it': 10,
        'N': 2,
        'M': 1,
        # 'track_list': ['update_norm', 'energy','x', 'consensus', 'drift'],
        'resampling': False,
        'update_thresh': 0.002}

#%% Define the objective function
mode = ''
if mode == 'import':
    f = Rastrigin()
else:
    # @cbx_objective_fh
    def f(x):
        x = torch.tensor(x)
        n = torch.linalg.norm(x, axis=-1)
        print(x)
        return n.numpy()

#%% Define the initial positions of the particles
x = cbx.utils.init_particles(shape=(conf['M'], conf['N'], conf['d']), x_min=-3., x_max = 3.)
# print('x', x.shape)
# x = torch.tensor(x)

#%% Define the CBO algorithm
dyn = CBO(f, x=x, noise='anisotropic', f_dim='3D', 
          **conf)

# dyn = CBO(f, x=x, **conf)


sched = scheduler(dyn, [multiply(name='alpha', factor=1.1, maximum=1e15),
                        #multiply(name='sigma', factor=1.005, maximum=6.)
                        ])
#%% Run the CBO algorithm
t = 0
it = 0
while not dyn.terminate():
    dyn.step()
    sched.update()
    # print('dyn.num_f_eval', dyn.num_f_eval)
    
    # if it%10 == 0:
    # print(dyn.f_min)
    # print('x', x)
    # print('x_best_particle_0', dyn.best_particle[0])
    print('****************')
    # print('x_best_cur_particle', dyn.best_cur_particle)
    # print('Alpha: ' + str(dyn.alpha))
    # print('Sigma: ' + str(dyn.sigma))
        
    it+=1
    
#%%
# plt.close('all')
# plotter = plot_dynamic(dyn, dims=[0,19], 
#                         contour_args={'x_min':-3, 'x_max':3},
#                         plot_consensus=True,
#                         plot_drift=True)
# plotter.run_plots(wait=0.05, freq=1)