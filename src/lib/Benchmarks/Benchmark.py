# SPDX-FileCopyrightText: The sinkhorn-rom authors, see file AUTHORS
#
# SPDX-License-Identifier: GPL-3.0-or-later

import numpy as np
from alive_progress import alive_bar
import time
from statistics import mean
import pickle
import matplotlib.pyplot as plt
plt.rcParams.update(plt.rcParamsDefault)
plt.rcParams["font.family"] = 'serif'
plt.rcParams["savefig.format"] = 'pdf'
# plt.rcParams['text.usetex'] = True
import seaborn as sns
sns.set_theme(style="darkgrid")
sns.set(rc={"figure.dpi":100, 'savefig.dpi':100})
sns.set_context("paper")
import torch
from ..DataManipulators.Problems import get_file_path
from ..DataManipulators.DataStruct import QueryDataStruct, TargetDataStruct
from ...config import results_dir, device
from ...visualization import get_colors
from ...utils import check_create_dir
from ..Evaluators.Barycenter import ImagesLoss, ImagesBarycenter_v2
from ..Evaluators.Image_reconstruction import computing_measure_attributes, reconstruct_target_w_static_sensors, optimize_over_Cs
    
# from ..Evaluators.Image_reconstruction_v2 import inf_stability_torch
# if use_pykeops:
#     from ..Evaluators.Barycenter_Implementation import logsumexp_template

import random
from ...visualization import plot_fields,plot_fields_images

def get_problem_dir(problem):
    return check_create_dir(results_dir+'{}/{}/'.format(problem.name, problem.id))

def get_fit_dir(problem, model):
    return check_create_dir(get_problem_dir(problem)+'{}/fit/{}/{}/'.format(model.name, problem.config_set_fit['nparam'], problem.config_set_fit['id_set']))

def get_predict_dir(problem, model):
    return check_create_dir(get_problem_dir(problem)+'{}/predict/{}/{}/'.format(model.name, problem.config_set_predict['nparam'], problem.config_set_predict['id_set']))

def get_predict_dir_v2(problem, model):
    return check_create_dir(get_problem_dir(problem)+'{}/predict/{}_{}/{}/'.format(model.name,problem.config_set_fit['nparam'], problem.config_set_predict['nparam'], problem.config_set_predict['id_set']))

#CLEAN:
# problems = [ Problem(name=args.p,
#                     id='Greedytest_burger2d_M64_ns100_nmax9_again',
#                     config_set_fit={'nparam': args.nfit, 'id_set': args.idfit},
#                     config_set_predict={'nparam': args.np, 'id_set': args.idp}),
# ]
# models = [
#     NonIntrusiveGreedyImages ( Loss = Loss, nmax = 9,
#                     compute_intermediate_interpolators = True,
#                     params_sinkhorn_bary=params_sinkhorn_bary,
#                     params_opt_best_barycenter=params_opt_best_barycenter),
# ]

# for problem in problems:
#     for model in models:
#         fit(problem, model)
#         predict(problem, model)
#         #pass
#     plots_fit(problem, models)
#     plots_predict(problem, models)


def fit(problem, model):
    """
        Fit a model for a given problem
    """

    # Training queries
    field_coordinates, snapshots, parameters, uuid = problem.load_dataset(**problem.config_set_fit)

    query_train = QueryDataStruct(parameter=parameters,field_spatial_coordinates=[field_coordinates])
    target_train = TargetDataStruct(field=snapshots)
    
    # Fit
    model.fit(query_train, target_train)

    # Save
    rootdir = get_fit_dir(problem, model)
    model.uuid_training = uuid # Save uuid of training set into the model to facilitate track of training set
    model.save_fit(rootdir)    # Save model
    
# def modify_fit(problem, model):
#     field_coordinates, snapshots, parameters, uuid = problem.load_dataset(**problem.config_set_fit)
#     query_train = QueryDataStruct(parameter=parameters,field_spatial_coordinates=[field_coordinates])
#     #target_train = TargetDataStruct(field=snapshots)
#     rootdir = get_fit_dir(problem, model)
#     interpolator_tools = pickle.load(open(rootdir+'barycentric_interpolator_tools', "rb" ))
#     print('param',interpolator_tools[0]['parameters'])
#     for id in range(len(interpolator_tools)):
#         interpolator_tools[id]['parameters'] = np.array([param.tolist() for param in query_train.parameter])
#     print('param_after',interpolator_tools[0]['parameters'])
#     pickle.dump(interpolator_tools, open(rootdir+'barycentric_interpolator_tools_new', "wb" ))

def state_estimation(problem, models):
    
    # Load test queries: Change **problem.config_set_fit based on launch file
    field_coordinates_train, snapshots_train, parameters_train, _ = problem.load_dataset(**problem.config_set_fit)
    
    field_coordinates, snapshots, parameters, _ = problem.load_dataset(**problem.config_set_predict)
        
    if field_coordinates is not torch.is_tensor(field_coordinates):
        field_coordinates = torch.tensor(field_coordinates, device=device)
    
    #Normalizing snapshots
    snaps_norm = list()
    for i in snapshots: snaps_norm.append(i/i.sum())

    #Load final barycenter from the Greedy algorithm
    model = models[0]
    dir_fit = get_fit_dir(problem, model)
    model.load_fit(dir_fit)
    barycenter = model.final_barycenter # in (B, K, N) format
    fit_output = model.fit_output
    
    # print('fit_output', len(fit_output))
    
    #Normalizing Greedy measures
    N =4 # Number of measures to for the barycenter
    D = fit_output[0].field[0].dim() # Spatial dimension of the problem
    if D == 1:
        measures_renorm = fit_output[0].field[0][:]/fit_output[0].field[0][:].sum()    
        measures_renorm = measures_renorm[None, None, :]
        for i in range(N): 
            tmp = fit_output[i+1].field[0][:]/fit_output[i+1].field[0][:].sum()
            measures_renorm = torch.cat((measures_renorm, tmp[None, None, :]), dim=1)
                        
    else:
        measures_renorm = fit_output[0].field[0]/fit_output[0].field[0].sum()    
        measures_renorm = measures_renorm[None, None, :, :]
        for i in range(N): 
            tmp = fit_output[i+1].field[0]/fit_output[i+1].field[0].sum()
            measures_renorm = torch.cat((measures_renorm, tmp[None, None, :, :]), dim=1)
            
    # measures_renorm.to(torch.device('cuda:0'))
    
    ######## 3 Translated measures #################
    #KdV1d[0]: 300; Burgers[0]: 400, 200;
    # Camassa-Holm[3]: 200, 400; Gaussian[0]: 100, 200
    # sigma = 0.1
    # gaussian = lambda x : (torch.exp(-0.5*((field_coordinates-x)/sigma)**2)/((sigma*(2*math.pi)**0.5)))
    # measures_renorm = fit_output[0].field[0][:]/fit_output[0].field[0][:].sum()
    # tmp1 = torch.roll(measures_renorm,  400)
    # tmp2 = torch.roll(measures_renorm,  200)
    
    # sigma = 0.1
    # gaussian  = lambda x : (torch.exp(-0.5*((field_coordinates-x)/sigma)**2)/((sigma*(2*math.pi)**0.5)))
    # tmp2      = torch.zeros(field_coordinates.squeeze().shape[0], device=device)
    # tmp1 = gaussian(-1).squeeze();  tmp1 = tmp1/tmp1.sum()
    # for i in range(tmp2.shape[0]):
    #     if field_coordinates.squeeze()[i] >= (1.-sigma) and field_coordinates.squeeze()[i] <= (1.+sigma):
    #         tmp2[i] = 1.0
            
    # tmp2 = tmp2/tmp2.sum()
    
    # measures_renorm = torch.cat((tmp1[None, None, :], tmp2[None, None, :]), dim=1)
    
    # measures_renorm = torch.cat((measures_renorm[None, None, :], tmp1[None, None, :]), dim=1)
    # measures_renorm = torch.cat((measures_renorm, tmp2[None, None, :]), dim=1)    
    
    ######### Choose a particular snapshot for evaluation #########

    # idx = random.sample(range(0,len(snaps_norm)), 1)
    # idx = [0] #KDV1d: 13, Burger1d: 22, Gaussian1d: 22
    # print('Chosen snapshot is fixed!')
    # print('Chosen snapshot from the pool', idx[0])
    
    # obs_target_field, obs_bary, bary, weights, L, _ = projObsGraFixSupp(snaps_norm[idx[0]][None, None, :]
    #                                                                     , measures_renorm, field_coordinates,
    #                                                                     problem.name)
    
    ######### Choose all snapshots for evaluation #########
    
    # print('All snapshots chosen from the pool !')
    
    # for i in range(len(snaps_norm)): snaps_norm[i] = snaps_norm[i][None, None, :] 
    
    # if D == 1:
    #     snaps_norm = [snaps_norm[0][None, None, :]]
    # else:
    #     snaps_norm = [snaps_norm[0][None, None, :, :]]
    
    # snaps_norm = snaps_norm[10][None, None, :] 
    
    # obs_target_field, obs_bary, bary, weights, L, _ = compute_stability_constant(snaps_norm, 
                                                                                 # measures_renorm, 
                                                                                 # field_coordinates, 
                                                                                 # problem.name)
                                                                                 
    # loss, evolution = sup_stability_const(measures_renorm, field_coordinates)

    # print('weights1:', evolution['weights1'])
    # print('weights2:', evolution['weights2'])
    # print('sup_Loss:', evolution['loss'])
    # print('(sup) C_s:', evolution['sup_stability_const'])    

    # inf_loss, inf_evolution, _ = inf_stability_const(snaps_norm[10][None, None, :],measures_renorm, field_coordinates, problem.name)
    # inf_loss, inf_evolution, _ = reconstruct_target_w_static_sensors(snaps_norm[10][None, None, :],measures_renorm, field_coordinates, problem.name)
    # inf_loss, inf_evolution = inf_stability_const(tmp[None, None, :],measures_renorm, field_coordinates, problem.name)
    
    optimize_over_Cs(tmp[None, None, :],measures_renorm, field_coordinates, problem.name)
    
    # obj_consensus_opt = consensus_opt(tmp[None, None, :],measures_renorm, field_coordinates, problem.name)
    # obj_consensus_opt = consensus_opt(snaps_norm[10][None, None, :],measures_renorm, field_coordinates, problem.name)
    # obj_consensus_opt.inf()
    # obj_consensus_opt.reconstruct_target_w_static_sensors()
    
    # frames_dir = results_dir+'Frames_compute_mu_'+problem.name+'/'
    # fig, axs = plt.subplots(2, sharex=True, sharey=True)
    
    # x = inf_evolution['iter']
    # inf_sup_C_s = inf_evolution['inf_sup_stability_const']
    # sup_C_s = inf_evolution['sup_stability_const']

    # axs[0].plot(x, sup_C_s)
    # axs[0].set_xlabel('Iterations')
    # axs[0].set_ylabel(r'$\sup_{\Lambda_1, \Lambda_2} \frac{W_2(\mu,\nu)}{||\ell(\mu)-\ell(\nu)||}$', rotation=90)
    
    # axs[1].plot(x, inf_sup_C_s)
    # axs[1].set_xlabel('Iterations')
    # axs[1].set_ylabel(r'$\inf_{x\in\Omega} \sup_{\Lambda_1, \Lambda_2} \frac{W_2(\mu,\nu)}{||\ell(\mu)-\ell(\nu)||}$', rotation=90)

    # fig.savefig(frames_dir+'C_s_'+problem.name)
    
    # Filename = frames_dir+'C_s_'+str(N+1)
    # file = open(Filename+'.txt', "w")
    # for i in inf_sup_C_s: file.write("%s\n" % i)
    # file.close()

    # fig = plt.figure()
    # # plt.plot(field_coordinates.numpy(), 
    # #           snaps_norm[idx[0]][:].numpy())
    # # plt.plot(field_coordinates.numpy(), 
    # #           barycenter.detach().squeeze().numpy())
    # plt.plot(field_coordinates.numpy(), 
    #           obs_target_field.detach().squeeze().numpy())
    # plt.plot(field_coordinates.numpy(), 
    #           obs_bary.detach().squeeze().numpy())
    # plt.set_xlabel('x')
    # plt.set_ylabel('Observables')
    # plt.legend([r'Target', 'Bary approx'],loc='best')
    # plt.show()
    
    # fig, axs = plt.subplots(1, sharex=True, sharey=True)
    # plt.plot(field_coordinates.numpy(), 
    #           snaps_norm[idx[0]][:].numpy())
    # plt.plot(field_coordinates.numpy(), 
    #           barycenter.detach().squeeze().numpy())
    
    # axs.plot(field_coordinates.numpy(),obs_target_field.detach().squeeze().numpy(), 'o')
    
    # axs.plot(field_coordinates.numpy(),obs_bary.detach().squeeze().numpy(), '*')    
    
    #Uncomment from here: ###
    
    # fig, axs = plt.subplots(1, sharex=True, sharey=True)

    # sensor_placement = torch.zeros(field_coordinates.shape[0])
    
    # axs.plot(field_coordinates.cpu().numpy(),snaps_norm[idx[0]].cpu().numpy())
    
    # axs.plot(field_coordinates.cpu().numpy(),bary.cpu().detach().squeeze().numpy())
        
    # axs.set_xlabel('x')
    # axs.set_ylabel(r'$\mu_{normalized}$', rotation=90)
    # axs.legend([r'Target', 'Bary approx'],loc='best')
    
    # fig.savefig(dir_fit+problem.name+'-nfit-{}'.format(problem.config_set_fit['nparam'])
    #             +'-idfit-{}'.format(problem.config_set_fit['id_set'])
    #             +'-np-{}'.format(problem.config_set_predict['nparam'])
    #             +'-idp-{}'.format(problem.config_set_predict['id_set']))
    
    # plt.close('all')
    
    #Uncomment until here: ###

def load_greedy_measures(problem, models):

    #Load predict snapshots
    field_coordinates, snapshots, parameters, _ = problem.load_dataset(**problem.config_set_predict)

    #Normalizing predict snapshots
    snaps_norm = list()
    for i in snapshots: snaps_norm.append(i/i.sum())

    #Load barycenter from the Greedy algorithm
    model = models[0]
    dir_fit = get_fit_dir(problem, model)
    model.load_fit(dir_fit)
    barycenter = model.final_barycenter # in (B, K, N) format
    fit_output = model.fit_output

    #Normalizing Greedy measures
    N =4 # Number of measures selecter from Greedy barycenter
    # D = fit_output[0].field[0].dim() # Spatial dimension of the problem
    # if D == 1:
    measures_renorm = fit_output[0].field[0][:]/fit_output[0].field[0][:].sum()    
    measures_renorm = measures_renorm[None, None, :]
    for i in range(N): 
        tmp = fit_output[i+1].field[0][:]/fit_output[i+1].field[0][:].sum()
        measures_renorm = torch.cat((measures_renorm, tmp[None, None, :]), dim=1)
                        
    # # else:
    #     measures_renorm = fit_output[0].field[0]/fit_output[0].field[0].sum()    
    #     measures_renorm = measures_renorm[None, None, :, :]
    #     for i in range(N): 
    #         tmp = fit_output[i+1].field[0]/fit_output[i+1].field[0].sum()
    #         # measures_renorm = torch.cat((measures_renorm, tmp[None, None, :, :]), dim=1)

    computing_measure_attributes(measures_renorm, field_coordinates)


def state_estimation_travelling_soln(problem, models):
    
    # Load test queries: Change **problem.config_set_fit based on launch file
    field_coordinates_train, snapshots_train, parameters_train, _ = problem.load_dataset(**problem.config_set_fit)
    
    field_coordinates, snapshots, parameters, _ = problem.load_dataset(**problem.config_set_predict)    
        
    if field_coordinates is not torch.is_tensor(field_coordinates):
        field_coordinates = torch.tensor(field_coordinates, device=device)
    
    #Normalizing snapshots
    snaps_norm = list()
    for i in snapshots: snaps_norm.append(i/i.sum())

    #Load final barycenter from the Greedy algorithm
    model = models[0]
    dir_fit = get_fit_dir(problem, model)
    model.load_fit(dir_fit)
    barycenter = model.final_barycenter # in (B, K, N) format
    fit_output = model.fit_output
    
    # print('fit_output', len(fit_output))
    
    #Normalizing Greedy measures
    N =4 # Number of measures to for the barycenter
    D = fit_output[0].field[0].dim() # Spatial dimension of the problem
    if D == 1:
        measures_renorm = fit_output[0].field[0][:]/fit_output[0].field[0][:].sum()    
        measures_renorm = measures_renorm[None, None, :]
        for i in range(N): 
            tmp = fit_output[i+1].field[0][:]/fit_output[i+1].field[0][:].sum()
            measures_renorm = torch.cat((measures_renorm, tmp[None, None, :]), dim=1)
                        
    else:
        measures_renorm = fit_output[0].field[0]/fit_output[0].field[0].sum()    
        measures_renorm = measures_renorm[None, None, :, :]
        for i in range(N): 
            tmp = fit_output[i+1].field[0]/fit_output[i+1].field[0].sum()
            measures_renorm = torch.cat((measures_renorm, tmp[None, None, :, :]), dim=1)
            
    ######## 3 Travelling solutions #################
    
    # Loading sensor locations and sensor arangements (central or evenly distributed)
    # load_dir = results_dir+'Compute_C_s_{}/five_measures/SGD/'.format(problem.name)
    load_dir = results_dir+'Compute_C_s_{}/five_measures/CBO/'.format(problem.name)
    arange   = 'even'
    # arange   = 'central'
    
    nsnaps = 11
    
    travelling_field_coord = []
    travelling_snapshot    = []
    travelling_param_train = []
    
    with alive_bar(nsnaps, bar='checks') as bar:
        for i in range(nsnaps):
            travelling_field_coord.append(torch.load(get_file_path(problem.name+'_transport/t{}/1/0/points'.format(i)), map_location=device))
            travelling_snapshot.append(torch.load(get_file_path(problem.name+'_transport/t{}/1/0/fields'.format(i)), map_location=device))
            travelling_param_train.append(torch.load(get_file_path(problem.name+'_transport/t{}/1/0/params'.format(i)), map_location=device))
            
            # DEBUG:
            # fig, axs = plt.subplots(1, sharex=True, sharey=True)
            # axs.plot(travelling_field_coord[i].detach().cpu().numpy(), travelling_snapshot[i][0].detach().cpu().numpy())
            # plt.show()
            # plt.close()
            
            time.sleep(.005)
            bar()
            
    print('######## Travelling solution loaded ########')
            
    with alive_bar(nsnaps, bar='classic') as bar:
        for i in range(10,11):
            
            snaps_norm = travelling_snapshot[i][0]/travelling_snapshot[i][0].sum()
                
            reconstruct_target_w_static_sensors(snaps_norm[None, None, :],
                                                  measures_renorm, field_coordinates,
                                                  problem.name, load_dir, arange, i)
            
            # inf_loss, inf_evolution, _ = inf_stability_const(travelling_snapshot[None, None, :],measures_renorm, field_coordinates, problem.name)
        
            # obj_consensus_opt = consensus_opt(snaps_norm[None, None, :],measures_renorm, field_coordinates, problem.name)
            # obj_consensus_opt.inf()
            # obj_consensus_opt.reconstruct_target_w_static_sensors(i,load_dir)
        
            # time.sleep(.005)
            bar()

    
def predict(problem, model, compute_err=True,visualize_samples=True):

    field_coordinates_train, _, parameters_train, _ = problem.load_dataset(**problem.config_set_fit)
    query_train = QueryDataStruct(parameter=parameters_train,field_spatial_coordinates=[field_coordinates_train])

    # min_max_scaler = preprocessing.MinMaxScaler()
    # min_max_scaler.fit(train_parameters.cpu())

    # Load test queries
    field_coordinates, snapshots, parameters, _ = problem.load_dataset(**problem.config_set_predict)

    # prep_parameters = min_max_scaler.transform(parameters.cpu())
    # prep_parameters = torch.tensor(prep_parameters, device=device)

    query_test = QueryDataStruct(parameter=parameters,field_spatial_coordinates=[field_coordinates])
    target_test = TargetDataStruct(field=snapshots)
    
    # DEBUG:
    # print('Spatial dim.',query_test[0].field_spatial_coordinates[0].dim())
    
    D = target_test[0].field[0].dim()

    # Load fit
    dir_fit = get_fit_dir(problem, model)
    model.load_fit(dir_fit)

    # prepare logsumexp if pykeops
    # if 'params_sinkhorn_bary' in model.__dict__ and use_pykeops:
    #     model.params_sinkhorn_bary['logsumexp'] = logsumexp_template(field_coordinates.shape[1], field_coordinates.shape[1])

    # Predict
    rootdir = get_predict_dir(problem, model)
    err = []
    if D == 1:
        for n in range(model.nmax):
            ext = '_{}_{}_{}'.format(problem.config_set_fit['nparam'], problem.config_set_predict['nparam'],n)
            predictions = model.predict(query_test, n=n)
            #DEBUG:
            # print('Data type of predictions', type(predictions))
            # print('Shape of elements in predictions', predictions[0].shape)
            error = [ np.sqrt(ImagesLoss(u.field[0][None,None,:], pred[None,None,:],blur=0.0001,scaling=0.9).item()) for (u, pred) in zip(target_test, predictions) ]
            err.append(error)
            # Save
            pickle.dump(error, open(rootdir+'err_predict'+ext, "wb" ))
            
            if visualize_samples:
                
                n_tests = len(query_test)
                num_predicts = 5
                #sample_ids = random.sample(np.arange(n_tests).tolist(), num_predicts)
                sample_ids = list(range(5))
                predictions = [model.core_predict(query_test[id],n=model.nmax-1) for id in sample_ids]
                for i in range(num_predicts):
                    my_fig_title = 'Approximation ' + str(sample_ids[i])
                    fig_opts={'colors': [], 'labels': ['ref', 'fit'], 'titles': None, 'plot_type': 'voronoi', 'rootdir': results_dir,  'fig_title': 'fields', 'format': '.pdf'}
                    plot_fields(fields= [target_test.field[sample_ids[i]], predictions[i].detach()],
                                spatial_coordinates= field_coordinates,
                                fig_opts=fig_opts,)
    else:
        for n in range(model.nmax):
            ext = '_{}_{}_{}'.format(problem.config_set_fit['nparam'], problem.config_set_predict['nparam'],n)
            predictions = model.predict(query_test, n=n)
            error = [ np.sqrt(ImagesLoss(u.field[0][None,None,:,:], pred[None,None,:,:],blur=0.001,scaling=0.9).item()) for (u, pred) in zip(target_test, predictions) ]
            err.append(error)
            # Save
            pickle.dump(error, open(rootdir+'err_predict'+ext, "wb" ))
            
            if visualize_samples:
                
                n_tests = len(query_test)
                num_predicts = 10
                #sample_ids = random.sample(np.arange(n_tests).tolist(), num_predicts)
                sample_ids = list(range(10))
                predictions = [model.core_predict(query_test[id],n=model.nmax-1) for id in sample_ids]
                for i in range(num_predicts):
                    my_fig_title = 'Approximation ' + str(sample_ids[i])
                    fig_opts={'colors': [], 'labels': ['ref', 'fit'], 'titles': None, 'plot_type': 'voronoi', 'rootdir': results_dir,  'fig_title': 'fields', 'format': '.pdf'}
                    plot_fields_images(fields= [target_test.field[sample_ids[i]].detach().cpu().numpy(), predictions[i].detach().cpu().numpy()],
                                spatial_coordinates= field_coordinates.cpu().numpy(),
                                fig_opts=fig_opts,)
    # pickle.dump(err, open(rootdir+'err_predict', "wb" ))


# def mypredict(problem, model,visualize_samples=True):

#     # _, _, train_parameters, _ = problem.load_dataset(**problem.config_set_fit)
#     # min_max_scaler = preprocessing.MinMaxScaler()
#     # min_max_scaler.fit(train_parameters.cpu())

#     # Load test queries
#     field_coordinates, snapshots, parameters, _ = problem.load_dataset(**problem.config_set_predict)

#     # prep_parameters = min_max_scaler.transform(parameters.cpu())
#     # prep_parameters = torch.tensor(prep_parameters, device=device)

#     query_test = QueryDataStruct(parameter=parameters,field_spatial_coordinates=[field_coordinates])
#     target_test = TargetDataStruct(field=snapshots)

#     # Load fit
#     dir_fit = get_fit_dir(problem, model)
#     model.load_fit(dir_fit)

#     # Predict
#     rootdir = get_predict_dir_v2(problem, model)
    
#     #predictions, weights, losses = model.weights_predict(query_test,target_test)
#     results = model.embedding_predict(query_test,target_test)

#     predictions = [ result['barycenter'] for result in results]
#     weights = [ result['weight'] for result in results]
#     indexes_KNN = [ result['index_KNN'] for result in results]


#     err = [ np.sqrt(ImagesLoss(u.field[0][None,None,:,:], pred[None,None,:,:],blur=0.001,scaling=0.9).item()) for (u, pred) in zip(target_test, predictions) ]
#     # Save
#     ext = '_{}_{}'.format(problem.config_set_fit['nparam'], problem.config_set_predict['nparam'])
#     pickle.dump(err, open(rootdir+'err_predict'+ext, "wb" ))
#     saveWeights = torch.cat([w[None,:].detach().cpu() for w in weights],dim=0)
#     saveIndexes_KNN = torch.vstack(indexes_KNN)
#     # print(saveIndexes_KNN)
#     # print('saveW',saveWeights)
#     # print('saveL',saveLosses)
#     torch.save(saveWeights, rootdir+'weights')
#     torch.save(saveIndexes_KNN, rootdir+'indexes_KNN')


#     if visualize_samples:
#         n_tests = len(query_test)
#         num_predicts = 5
#         sample_ids = random.sample(np.arange(n_tests).tolist(), num_predicts)
#         sample_predictions = [predictions[id] for id in sample_ids]
#         for i in range(num_predicts):
#             my_fig_title = 'Approximation ' + str(sample_ids[i])
#             fig_opts={'colors': [], 'labels': ['ref', 'fit'], 'titles': None, 'plot_type': 'voronoi', 'rootdir': rootdir,  'fig_title': my_fig_title, 'format': '.pdf'}
#             plot_fields_images(fields= [target_test.field[sample_ids[i]].detach().cpu().numpy(), sample_predictions[i].detach().cpu().numpy()],
#                         spatial_coordinates= field_coordinates.cpu().numpy(),
#                         fig_opts=fig_opts,)


def predict_weight_Greedy(problem, model,visualize_samples=True):

    # _, _, train_parameters, _ = problem.load_dataset(**problem.config_set_fit)
    # min_max_scaler = preprocessing.MinMaxScaler()
    # min_max_scaler.fit(train_parameters.cpu())

    # Load test queries
    field_coordinates, snapshots, parameters, _ = problem.load_dataset(**problem.config_set_predict)
    # prep_parameters = min_max_scaler.transform(parameters.cpu())
    # prep_parameters = torch.tensor(prep_parameters, device=device)
    query_test = QueryDataStruct(parameter=parameters,field_spatial_coordinates=[field_coordinates])
    target_test = TargetDataStruct(field=snapshots)

    # Load fit
    dir_fit = get_fit_dir(problem, model)
    model.load_fit(dir_fit)


    rootdir = get_predict_dir(problem, model)
    results = model.weights_predict(query_test,target_test)

    predictions = [ result['barycenter'] for result in results]
    weights = [ result['weight'] for result in results]
    losses = [ result['loss'] for result in results]
    

    err = [ np.sqrt(ImagesLoss(u.field[0][None,None,:,:], pred[None,None,:,:],blur=0.001,scaling=0.9).item()) for (u, pred) in zip(target_test, predictions) ]
    # Save
    ext = '_{}_{}'.format(problem.config_set_fit['nparam'], problem.config_set_predict['nparam'])
    pickle.dump(err, open(rootdir+'err_predict'+ext, "wb" ))
    saveWeights = torch.cat([w[None,:].detach().cpu() for w in weights],dim=0)
    saveLosses = torch.tensor(losses)
    
    torch.save(saveWeights, rootdir+'weights')
    torch.save(saveLosses, rootdir+'losses')
    if visualize_samples:
        n_tests = len(query_test)
        num_predicts = 5
        sample_ids = random.sample(np.arange(n_tests).tolist(), num_predicts)
        sample_predictions = [predictions[id] for id in sample_ids]
        for i in range(num_predicts):
            my_fig_title = 'Approximation ' + str(sample_ids[i])
            fig_opts={'colors': [], 'labels': ['ref', 'fit'], 'titles': None, 'plot_type': 'voronoi', 'rootdir': rootdir,  'fig_title': my_fig_title, 'format': '.pdf'}
            # plot_fields(fields= [target_test.field[sample_ids[i]].detach().cpu().numpy(), sample_predictions[i].detach().cpu().numpy()],
            #             spatial_coordinates= field_coordinates.cpu().numpy(),
            #             fig_opts=fig_opts,)
            plot_fields_images(fields= [target_test.field[sample_ids[i]].detach().cpu().numpy(), sample_predictions[i].detach().cpu().numpy()],
                        spatial_coordinates= field_coordinates.cpu().numpy(),
                        fig_opts=fig_opts,)
    





# def predict_mds(problem, model, compute_time=False,visualize_samples=True):
#     # prameters train
#     _, _, train_parameters, _ = problem.load_dataset(**problem.config_set_fit)
#     min_max_scaler = preprocessing.MinMaxScaler()
#     min_max_scaler.fit(train_parameters.cpu())

#     # Load test queries
#     field_coordinates, snapshots, parameters, _ = problem.load_dataset(**problem.config_set_predict)
#     prep_parameters = min_max_scaler.transform(parameters.cpu())
#     prep_parameters = torch.tensor(prep_parameters, device=device)

#     query_test = QueryDataStruct(parameter=prep_parameters,field_spatial_coordinates=[field_coordinates])
#     target_test = TargetDataStruct(field=snapshots)

#     # Load fit
#     dir_fit = get_fit_dir(problem, model)
#     model.load_fit(dir_fit)
#     # Predict
#     rootdir = get_predict_dir_v2(problem, model)
#     if compute_time:
#         predictions, prediction_times = model.predict_times(query_test)
#         pickle.dump(prediction_times, open(rootdir + 'prediction_times', "wb"))
#     else:
#         predictions = model.predict(query_test)
#     #err = [ model.Loss(u.field[0].flatten(), field_coordinates, pred.flatten(), field_coordinates).item() for (u, pred) in zip(target_test, predictions) ]
#     err = [ np.sqrt(ImagesLoss(u.field[0][None,None,:,:], pred[None,None,:,:],blur=0.001,scaling=0.9).item()) for (u, pred) in zip(target_test, predictions) ]
#     # Save
#     pickle.dump(err, open(rootdir+'err_predict', "wb" ))
#     if visualize_samples:
#         n_tests = len(query_test)
#         num_predicts = 5
#         sample_ids = random.sample(np.arange(n_tests).tolist(), num_predicts)
#         predictions = [model.core_predict(query_test[id])[0] for id in sample_ids]
#         for i in range(num_predicts):
#             my_fig_title = 'Approximation ' + str(sample_ids[i])
#             fig_opts={'colors': [], 'labels': ['ref', 'fit'], 'titles': None, 'plot_type': 'voronoi', 'rootdir': rootdir,  'fig_title': my_fig_title, 'format': '.pdf'}
#             plot_fields_v2(fields= [target_test.field[sample_ids[i]].detach().cpu().numpy(), predictions[i].detach().cpu().numpy()],
#                         spatial_coordinates= field_coordinates.cpu().numpy(),
#                         fig_opts=fig_opts,)


# def predict_images(problem, model, compute_time=False,visualize_samples=True):

#     _, _, train_parameters, _ = problem.load_dataset(**problem.config_set_fit)
#     min_max_scaler = preprocessing.MinMaxScaler()
#     min_max_scaler.fit(train_parameters.cpu())

#     # Load test queries
#     field_coordinates, snapshots, parameters, _ = problem.load_dataset(**problem.config_set_predict)

#     prep_parameters = min_max_scaler.transform(parameters.cpu())
#     prep_parameters = torch.tensor(prep_parameters, device=device)

#     query_test = QueryDataStruct(parameter=prep_parameters,
#                                 field_spatial_coordinates=[field_coordinates])
#     target_test = TargetDataStruct(field=snapshots)

#     # Load fit
#     dir_fit = get_fit_dir(problem, model)
#     model.load_fit(dir_fit)

#     # Predict
#     rootdir = get_predict_dir_v2(problem, model)
#     if compute_time:
#         predictions, prediction_times = model.predict_times(query_test)
#         pickle.dump(prediction_times, open(rootdir + 'prediction_times', "wb"))
#     else:
#         predictions = model.predict(query_test)
#     #err = [ model.Loss(u.field[0].flatten(), field_coordinates, pred.flatten(), field_coordinates).item() for (u, pred) in zip(target_test, predictions) ]
#     err = [ np.sqrt(ImagesLoss(u.field[0][None,None,:,:], pred[None,None,:,:],blur=0.001,scaling=0.9).item()) for (u, pred) in zip(target_test, predictions) ]
#     # Save
#     ext = '_{}_{}'.format(problem.config_set_fit['nparam'], problem.config_set_predict['nparam'])
#     pickle.dump(err, open(rootdir+'err_predict'+ext, "wb" ))



#     if visualize_samples:
#         n_tests = len(query_test)
#         num_predicts = 5
#         sample_ids = random.sample(np.arange(n_tests).tolist(), num_predicts)
#         sample_predictions = [predictions[id] for id in sample_ids]
#         for i in range(num_predicts):
#             my_fig_title = 'Approximation ' + str(sample_ids[i])
#             fig_opts={'colors': [], 'labels': ['ref', 'fit'], 'titles': None, 'plot_type': 'voronoi', 'rootdir': rootdir,  'fig_title': my_fig_title, 'format': '.pdf'}
#             plot_fields_images(fields= [target_test.field[sample_ids[i]].detach().cpu().numpy(), sample_predictions[i].detach().cpu().numpy()],
#                         spatial_coordinates= field_coordinates.cpu().numpy(),
#                         fig_opts=fig_opts,)


# def sample_predict_images(problem, model):

#     _, _, train_parameters, _ = problem.load_dataset(**problem.config_set_fit)
#     min_max_scaler = preprocessing.MinMaxScaler()
#     min_max_scaler.fit(train_parameters.cpu())

#     # Load test queries
#     field_coordinates, snapshots, parameters, _ = problem.load_dataset(**problem.config_set_predict)

#     prep_parameters = min_max_scaler.transform(parameters.cpu())
#     prep_parameters = torch.tensor(prep_parameters, device=device)

#     query_test = QueryDataStruct(parameter=prep_parameters,
#                                 field_spatial_coordinates=[field_coordinates])
#     target_test = TargetDataStruct(field=snapshots)

#     # Load fit
#     dir_fit = get_fit_dir(problem, model)
#     model.load_fit(dir_fit)

#     # Predict
#     rootdir = get_predict_dir_v2(problem, model)


#     n_tests = len(query_test)
#     num_predicts = 5
#     #sample_ids = random.sample(np.arange(n_tests).tolist(), num_predicts)
#     sample_ids  = [10,20,50,100,181]
#     predictions = [model.core_predict(query_test[id]) for id in sample_ids]
#     for i in range(len(sample_ids)):
#         my_fig_title = 'Approximation ' + str(sample_ids[i])
#         fig_opts={'colors': [], 'labels': ['ref', 'fit'], 'titles': None, 'plot_type': 'voronoi', 'rootdir': rootdir,  'fig_title': my_fig_title, 'format': '.pdf'}
#         plot_fields_images(fields= [target_test.field[sample_ids[i]].detach().cpu().numpy(), predictions[i].detach().cpu().numpy()],
#                     spatial_coordinates= field_coordinates.cpu().numpy(),
#                     fig_opts=fig_opts,)


# def compare_weights(problem, model):
#     # Load test queries
#     field_coordinates, snapshots, parameters, _ = problem.load_dataset(**problem.config_set_predict)
#     query_test = QueryDataStruct(parameter=parameters,
#                                 field_spatial_coordinates=[field_coordinates])
#     target_test = TargetDataStruct(field=snapshots)

#     # Load fit
#     dir_fit = get_fit_dir(problem, model)
#     model.load_fit(dir_fit)


#     # Predict
#     rootdir = get_predict_dir_v2(problem, model)

#     weights, best_weights = model.weights_predict(query_test,target_test)

#     # err = [ np.sqrt(ImagesLoss(u.field[0][None,None,:,:], pred[None,None,:,:],blur=0.001,scaling=0.9).item()) for (u, pred) in zip(target_test, predictions) ]
#     # pickle.dump(err, open(rootdir+'err_predict', "wb" ))



#     # if visualize_samples:
#     #     n_tests = len(query_test)
#     #     num_predicts = 5
#     #     sample_ids = random.sample(np.arange(n_tests).tolist(), num_predicts)
#     #     predictions = [model.core_predict(query_test[id]) for id in sample_ids]
#     #     for i in range(num_predicts):
#     #         my_fig_title = 'Approximation ' + str(sample_ids[i])
#     #         fig_opts={'colors': [], 'labels': ['ref', 'fit'], 'titles': None, 'plot_type': 'voronoi', 'rootdir': rootdir,  'fig_title': my_fig_title, 'format': '.pdf'}
#     #         plot_fields_images(fields= [target_test.field[sample_ids[i]].detach().cpu().numpy(), predictions[i].detach().cpu().numpy()],
#     #                     spatial_coordinates= field_coordinates.cpu().numpy(),
#     #                     fig_opts=fig_opts,)




# def knn_predict_images(problem, model,visualize_samples=True):

#     _, _, train_parameters, _ = problem.load_dataset(**problem.config_set_fit)
#     min_max_scaler = preprocessing.MinMaxScaler()
#     min_max_scaler.fit(train_parameters.cpu())

#     # Load test queries
#     field_coordinates, snapshots, parameters, _ = problem.load_dataset(**problem.config_set_predict)

#     prep_parameters = min_max_scaler.transform(parameters.cpu())
#     prep_parameters = torch.tensor(prep_parameters, device=device)

#     query_test = QueryDataStruct(parameter=prep_parameters,
#                                 field_spatial_coordinates=[field_coordinates])
#     target_test = TargetDataStruct(field=snapshots)

#     # Load fit
#     dir_fit = get_fit_dir(problem, model)
#     model.load_fit(dir_fit)

#     # Predict
#     rootdir = get_predict_dir_v2(problem, model)
    
#     predictions = model.knn_predict(query_test,target_test)
#     err = [ np.sqrt(ImagesLoss(u.field[0][None,None,:,:], pred[None,None,:,:],blur=0.001,scaling=0.9).item()) for (u, pred) in zip(target_test, predictions) ]
#     # Save
#     ext = '_{}_{}'.format(problem.config_set_fit['nparam'], problem.config_set_predict['nparam'])
#     pickle.dump(err, open(rootdir+'err_predict'+ext, "wb" ))



#     if visualize_samples:
#         n_tests = len(query_test)
#         num_predicts = 5
#         sample_ids = random.sample(np.arange(n_tests).tolist(), num_predicts)
#         sample_predictions = [predictions[id] for id in sample_ids]
#         for i in range(num_predicts):
#             my_fig_title = 'Approximation ' + str(sample_ids[i])
#             fig_opts={'colors': [], 'labels': ['ref', 'fit'], 'titles': None, 'plot_type': 'voronoi', 'rootdir': rootdir,  'fig_title': my_fig_title, 'format': '.pdf'}
#             plot_fields_images(fields= [target_test.field[sample_ids[i]].detach().cpu().numpy(), sample_predictions[i].detach().cpu().numpy()],
#                         spatial_coordinates= field_coordinates.cpu().numpy(),
#                         fig_opts=fig_opts,)

def predict_weights(problem, model,visualize_samples=True):

    # Load test queries
    field_coordinates, snapshots, parameters, _ = problem.load_dataset(**problem.config_set_predict)
    query_test = QueryDataStruct(parameter=parameters,field_spatial_coordinates=[field_coordinates])
    target_test = TargetDataStruct(field=snapshots)

    # Load fit
    dir_fit = get_fit_dir(problem, model)
    model.load_fit(dir_fit)

    # Predict
    rootdir = get_predict_dir_v2(problem, model)
    results = model.weights_predict(query_test,target_test)

    predictions = [ result['barycenter'] for result in results]

    weights = [ result['weight'] for result in results]
    saveWeights = torch.cat([w[None,:].detach().cpu() for w in weights],dim=0)
    torch.save(saveWeights, rootdir+'weights')

    losses = [ result['loss'] for result in results]
    saveLosses = torch.tensor(losses)
    torch.save(saveLosses, rootdir+'losses')

    indexes_KNN = [ result['index_KNN'] for result in results]
    saveIndexes_KNN = torch.vstack(indexes_KNN)
    torch.save(saveIndexes_KNN, rootdir+'indexes_KNN')
    

    err = [ np.sqrt(ImagesLoss(u.field[0][None,None,:,:], pred[None,None,:,:],blur=0.001,scaling=0.9).item()) for (u, pred) in zip(target_test, predictions) ]
    # Save
    ext = '_{}_{}'.format(problem.config_set_fit['nparam'], problem.config_set_predict['nparam'])
    pickle.dump(err, open(rootdir+'err_predict'+ext, "wb" ))



    if visualize_samples:
        n_tests = len(query_test)
        num_predicts = 5
        sample_ids = random.sample(np.arange(n_tests).tolist(), num_predicts)
        sample_predictions = [predictions[id] for id in sample_ids]
        for i in range(num_predicts):
            my_fig_title = 'Approximation ' + str(sample_ids[i])
            fig_opts={'colors': [], 'labels': ['ref', 'fit'], 'titles': None, 'plot_type': 'voronoi', 'rootdir': rootdir,  'fig_title': my_fig_title, 'format': '.pdf'}
            plot_fields_images(fields= [target_test.field[sample_ids[i]].detach().cpu().numpy(), sample_predictions[i].detach().cpu().numpy()],
                        spatial_coordinates= field_coordinates.cpu().numpy(),
                        fig_opts=fig_opts,)

def sample_visualization(problem,model):
    # Load test queries
    field_coordinates, snapshots, parameters, _ = problem.load_dataset(**problem.config_set_predict)
    query_test = QueryDataStruct(parameter=parameters,field_spatial_coordinates=[field_coordinates])
    target_test = TargetDataStruct(field=snapshots)

    # Load fit
    dir_fit = get_fit_dir(problem, model)
    model.load_fit(dir_fit)

    # Predict
    rootdir = get_predict_dir_v2(problem, model)
    visualization_dir = check_create_dir(rootdir+ 'sample/')
    weights = torch.load(rootdir+'weights')
    indexes_KNN = torch.load(rootdir+'indexes_KNN')
    snapshots_train = model.interpolator_tools['snapshots'].to(device)
    # print('weights',weights)
    # print('indexes',indexes_KNN)
    sample_ids = list(range(10))
    for id in sample_ids:
        weight = weights[id][None,:].to(device=device)
        indices = indexes_KNN[id]

        measures = torch.cat([snapshots_train[indices[i]][None,None,:,:] for i in range(indices.shape[0])], dim=1)
        barycenter  = ImagesBarycenter_v2(measures=measures, weights=weight,**model.params_sinkhorn_bary)
        torch.save(barycenter[0,0], visualization_dir+'barycenter_' + str(id))
        torch.save(target_test.field[id], visualization_dir+'target_'+ str(id))
        my_fig_title = 'Approximation ' + str(id)
        fig_opts={'colors': [], 'labels': ['ref', 'fit'], 'titles': None, 'rootdir': visualization_dir,  'fig_title': my_fig_title, 'format': '.pdf'}
        plot_fields_images(fields= [target_test.field[id].detach().cpu().numpy(), barycenter[0,0].detach().cpu().numpy()],spatial_coordinates= field_coordinates.cpu().numpy(),fig_opts=fig_opts,)
        # save bary
        bary_fig_title = 'Barycenter ' + str(id)
        bary_fig_opts={'colors': [], 'labels': ['fit'],'titles': None, 'rootdir': visualization_dir,  'fig_title': bary_fig_title, 'format': '.pdf'}
        plot_fields_images(fields= [barycenter[0,0].detach().cpu().numpy()],spatial_coordinates= field_coordinates.cpu().numpy(),fig_opts=bary_fig_opts,)
        # save target
        target_fig_title = 'Target ' + str(id)
        target_fig_opts={'colors': [], 'labels': ['ref'],'titles': None, 'rootdir': visualization_dir,  'fig_title': target_fig_title, 'format': '.pdf'}
        plot_fields_images(fields= [target_test.field[id].detach().cpu().numpy()],spatial_coordinates= field_coordinates.cpu().numpy(),fig_opts=target_fig_opts,)











# def analysis_weights(problem, model):

#     field_coordinates_basis, snapshots_basis, parameters_basis, _ = problem.load_dataset(**problem.config_set_fit)
#     # min_max_scaler = preprocessing.MinMaxScaler()
#     # min_max_scaler.fit(train_parameters.cpu())

#     # Load train queries
#     field_coordinates_train, snapshots_train, parameters_train, _ = problem.load_dataset(**problem.config_set_predict)

#     # prep_parameters = min_max_scaler.transform(parameters.cpu())
#     # prep_parameters = torch.tensor(prep_parameters, device=device)

    
#     parameters_basis_train = torch.vstack((parameters_basis,parameters_train))
#     snapshots_basis_train  = snapshots_basis+snapshots_train

    
#     # print(parameters_basis_train.shape)
#     # print(snapshots_basis[0])
#     # print(snapshots_basis_train[0])
#     # print(snapshots_train[0])
#     # print(snapshots_basis_train[100])

#     query_train = QueryDataStruct(parameter=parameters_train,field_spatial_coordinates=[field_coordinates_train])
#     target_train = TargetDataStruct(field=snapshots_train)

#     query_basis_train = QueryDataStruct(parameter=parameters_basis_train,field_spatial_coordinates=[field_coordinates_basis])
#     target_basis_train = TargetDataStruct(field=snapshots_basis_train)

#     # Load test
#     field_coordinates_test, snapshots_test, parameters_test, _ = problem.load_dataset(nparam=100,id_set=2)
#     query_test = QueryDataStruct(parameter=parameters_test,field_spatial_coordinates=[field_coordinates_test])
#     target_test = TargetDataStruct(field=snapshots_test)

#     # Load fit
#     dir_fit = get_fit_dir(problem, model)
#     model.load_fit(dir_fit)

#     # Predict
#     rootdir_train = get_predict_dir_v2(problem, model)
#     matdir_train = check_create_dir(rootdir_train+'/matrix/')
    
#     weights_basis = torch.eye(parameters_basis.shape[0])
#     weights_train=torch.load(rootdir_train+'weights')
#     losses_train=torch.load(rootdir_train+'losses')
#     indexes_train = [torch.nonzero(weights_train[id]).flatten() for id in range(weights_train.shape[0])]

#     weights_basis_train = torch.vstack((weights_basis,weights_train))
#     indexes_basis_train = [torch.nonzero(weights_basis_train[id]).flatten() for id in range(weights_basis_train.shape[0])]
#     print(weights_basis_train.shape)
#     #print(indexes_basis_train[0:3])



#     # support_train = torch.hstack(indexes_train)
#     # df = pd.DataFrame(support.numpy(),columns=['index'])
#     # id_count = df['index'].value_counts()
#     # print(id_count.head(40))
#     # #ax = df.plot.hist(bins=100, alpha=0.5)
#     # #plt.show()

#     # Compute the distance matrix
#     #distance_matrix, pairwise_distance_tool = get_dissimilarity_matrix(target_val)
#     # print('DM',distance_matrix)
#     # print(pairwise_distance_tool['distance'])
#     # print(pairwise_distance_tool['firstIndex'])
#     # print(pairwise_distance_tool['secondIndex'])

#     # Analysis 
#     errors =[]
#     predictions =[]
#     rootdir_test = check_create_dir(get_problem_dir(problem)+'{}/predict/{}_{}/{}/'.format(model.name,problem.config_set_fit['nparam'],problem.config_set_predict['nparam'], 2))
#     supportdir_test = check_create_dir(rootdir_test+'/support_knn10_random/')
#     weights_test=torch.load(rootdir_test+'weights')
#     losses_test=torch.load(rootdir_test+'losses')
#     indexes_test = [torch.nonzero(weights_test[id]).flatten() for id in range(weights_test.shape[0])]
    

#     n_tests = len(query_test)
#     num_predicts = 100
#     #sample_ids = random.sample(np.arange(n_tests).tolist(), num_predicts)
#     sample_ids = [id for id in range(num_predicts)]
    
#     for i in range(num_predicts):
#         si = sample_ids[i]
#         # Find nearest q's neighbors
#         # KNN = NearestNeighbors(n_neighbors=4, algorithm='ball_tree').fit(parameters.cpu())
#         # param_distances, param_index_neighbors = KNN.kneighbors([query_test[si].parameter[0].tolist()])
#         # param_knn_index_neighbors = [indexes[id] for id in param_index_neighbors.flatten()]
#         # print('param_knn',param_knn_index_neighbors)
#         # param_knn_weight =[ total_weights[id][indexes[id]] for id in param_index_neighbors.flatten()]
#         # print('param_knn_weight',param_knn_weight)
#         # param_knn_list = []
#         # for id in param_index_neighbors.flatten():
#         #     param_knn_list.extend(indexes[id].tolist())
#         # param_counter = Counter(param_knn_list)
#         # print('param_counter',param_counter)


#         # visulaization
#         # param_unique_index_neighbors = torch.unique(torch.hstack(param_knn_index_neighbors),sorted=True)
#         # #print(param_unique_index_neighbors)
#         # param_neighbor_distance_matrix = restriction_distance_matrix(distance_matrix,param_unique_index_neighbors)
#         # #print(param_neighbor_distance_matrix)
#         # fig_opts={'colors': [], 'labels': None, 'titles': None, 'rootdir': matrix_dir,  'fig_title': 'param'+str(si), 'format': '.pdf'}
#         # fig,ax = plt.subplots()
#         # im=plt.imshow(param_neighbor_distance_matrix)
#         # ax.invert_yaxis() 
#         # fig.colorbar(im)
#         # fig.savefig(fig_opts['rootdir'] + fig_opts['fig_title'] + fig_opts['format'])
#         # #plt.show()

#         # Find nearest neighbors on snapshots space
#         snapshot_distances, snapshot_index_neighbors = get_KNN_snapshots(target_basis_train,target_test[si],n_neighbors=10)
#         snapshot_knn_index_neighbors = [indexes_basis_train[id] for id in snapshot_index_neighbors.flatten()]
#         print('snapshot_knn',snapshot_knn_index_neighbors)
#         snapshot_knn_weight =[ weights_basis_train[id][indexes_basis_train[id]] for id in snapshot_index_neighbors.flatten()]
#         print('snapshot_knn_weight',snapshot_knn_weight)
#         print('snapshot_distances',snapshot_distances)
#         # inverse weight distance
#         min_val = torch.min(snapshot_distances)
#         idx = torch.argmin(snapshot_distances)
#         if min_val <= 1.e-5:
#             print('close point')
#             alpha = torch.zeros(snapshot_distances.shape[0],device=device,dtype=torch.float64)
#             alpha[idx] = 1.0
#         else:
#             alpha = 1./snapshot_distances**1
#             alpha = alpha/torch.sum(alpha)
#             alpha= alpha.to(dtype=torch.float64)
#         #print('alpha',alpha)
#         full_snapshot_knn_weight = weights_basis_train[snapshot_index_neighbors.flatten()].to(device=device)
        
#         weights = torch.matmul(alpha[None,:],full_snapshot_knn_weight)
#         #print('weight', weights)
#         isFixSupport = True
#         n_keep = 10
#         if isFixSupport:
#             index_sort= torch.argsort(weights,dim=1,descending=True).flatten()
#             index_keep = index_sort[:n_keep]
#             index_zero = index_sort[n_keep:]
#             weights[:,index_zero]= 0.
#             # projection on the simplex
#             weights[:,index_keep] = mat2simplex(weights[:,index_keep])
#         S_index= torch.nonzero(torch.where(weights > 1.e-4, weights, 0.).flatten()).flatten()
#         print('S_index',S_index)
#         Rweights = weights[:,S_index]
        
#         snapshot_knn_list = []
#         for id in snapshot_index_neighbors.flatten():
#             snapshot_knn_list.extend(indexes_basis_train[id].tolist())

#         snapshot_counter = Counter(snapshot_knn_list)
#         print('snapshot_counter',snapshot_counter)
#         print('indexes_test',indexes_test[si])


#         # Interpolation  
#         # fields = [snapshots_basis[id] for id in S_index]
#         # measures = torch.cat([field[None,None,:,:] for field in fields], dim=1)
#         # bary = ImagesBarycenter_v2(measures=measures,weights=Rweights, **model.params_sinkhorn_bary)

#         # KNN
#         myfields = [snapshots_basis_train[id] for id in snapshot_index_neighbors.flatten()]
#         mymeasures = torch.cat([field[None,None,:,:] for field in myfields], dim=1)
#         #myweights = alpha[None,:]
#         # uniform weights
#         #myweights = torch.ones((1,snapshot_distances.shape[0]), device=device)/snapshot_distances.shape[0]
#         # random weights
#         myweights = torch.rand((1,snapshot_distances.shape[0]),device=device)
#         myweights.data = mat2simplex(myweights)
#         print('myweights',myweights)
#         bary = ImagesBarycenter_v2(measures=mymeasures,weights=myweights, **model.params_sinkhorn_bary)

#         # Nearest Neights
#         # Nearest = snapshots_basis_train[snapshot_index_neighbors[0]]
#         # bary = Nearest[None,None,:,:]

#         err = ImagesLoss(target_test[si].field[0][None,None,:,:], bary,blur=0.0001,scaling=0.9).item()
#         print('err',err)
#         print('loss_test',losses_test[si])
#         errors.append(err)
#         predictions.append(bary)
#         # weights
#         # weights_fig_title = 'weights ' + str(sample_ids[i])
#         # weights_fig_opts={'colors': [], 'labels': ['ref', 'fit'], 'titles': None, 'plot_type': 'voronoi', 'rootdir': supportdir_test,  'fig_title': weights_fig_title, 'format': '.pdf'}
#         # plot_fields_weights(fields=[weights_test[si].detach().cpu().numpy(), weights.detach().cpu().numpy()],fig_opts=weights_fig_opts)
#         ## snapshot
#         my_fig_title = 'Approximation ' + str(sample_ids[i])
#         fig_opts={'colors': [], 'labels': ['ref', 'fit'], 'titles': None, 'plot_type': 'voronoi', 'rootdir': supportdir_test,  'fig_title': my_fig_title, 'format': '.pdf'}
#         plot_fields_images(fields= [target_test.field[sample_ids[i]].detach().cpu().numpy(), bary[0,0].detach().cpu().numpy()],
#                         spatial_coordinates= field_coordinates_test.cpu().numpy(),fig_opts=fig_opts,)


#         # Visualization
#         # snapshot_unique_index_neighbors = torch.unique(torch.hstack(snapshot_knn_index_neighbors),sorted=True)
#         # #print(snapshot_unique_index_neighbors)
#         # snapshot_neighbor_distance_matrix = restriction_distance_matrix(distance_matrix,snapshot_unique_index_neighbors)
#         # #print(snapshot_neighbor_distance_matrix)

#         # fig_opts={'colors': [], 'labels': None, 'titles': None, 'rootdir': matrix_dir,  'fig_title': 'snapshot'+str(si), 'format': '.pdf'}
#         # fig, ax = plt.subplots()
#         # im = plt.imshow(snapshot_neighbor_distance_matrix)
#         # ax.invert_yaxis() 
#         # fig.colorbar(im)
#         # fig.savefig(fig_opts['rootdir'] + fig_opts['fig_title'] + fig_opts['format'])
#         # #plt.show()

#     print('errors',errors)
#     print('ex_errors',losses_test[sample_ids])
#     # save errors
#     np.save(supportdir_test+'fit_error',np.array(errors),allow_pickle=True)
#     np.save(supportdir_test+'ref_error',np.array(losses_test[sample_ids]),allow_pickle=True)

#     fig = plt.figure()
#     plt.plot(np.arange(len(errors)),losses_test[sample_ids],color='r', linestyle='-',marker= 'o', linewidth=2)
#     plt.plot(np.arange(len(errors)),errors,color='b', linestyle='-',marker= '*', linewidth=2)
#     plt.legend(['target','fit'])
#     plt.yscale('log')
#     plt.savefig(supportdir_test + 'loss.pdf')
#     plt.close()
    
##==============================================================================================================================
# def analysis_weights_knn(problem, model):

#     field_coordinates_basis, snapshots_basis, parameters_basis, _ = problem.load_dataset(**problem.config_set_fit)
#     # min_max_scaler = preprocessing.MinMaxScaler()
#     # min_max_scaler.fit(train_parameters.cpu())

#     # Load train queries
#     field_coordinates_train, snapshots_train, parameters_train, _ = problem.load_dataset(**problem.config_set_predict)

#     # prep_parameters = min_max_scaler.transform(parameters.cpu())
#     # prep_parameters = torch.tensor(prep_parameters, device=device)

    
#     parameters_basis_train = torch.vstack((parameters_basis,parameters_train))
#     snapshots_basis_train  = snapshots_basis+snapshots_train


#     query_train = QueryDataStruct(parameter=parameters_train,field_spatial_coordinates=[field_coordinates_train])
#     target_train = TargetDataStruct(field=snapshots_train)

#     query_basis_train = QueryDataStruct(parameter=parameters_basis_train,field_spatial_coordinates=[field_coordinates_basis])
#     target_basis_train = TargetDataStruct(field=snapshots_basis_train)

#     # Load test
#     field_coordinates_test, snapshots_test, parameters_test, _ = problem.load_dataset(nparam=100,id_set=2)
#     query_test = QueryDataStruct(parameter=parameters_test,field_spatial_coordinates=[field_coordinates_test])
#     target_test = TargetDataStruct(field=snapshots_test)

#     # Load fit
#     dir_fit = get_fit_dir(problem, model)
#     model.load_fit(dir_fit)

#     # Predict
#     rootdir_train = get_predict_dir_v2(problem, model)
#     matdir_train = check_create_dir(rootdir_train+'/matrix/')
    

#     # Analysis 
#     errors =[]
#     predictions =[]
#     rootdir_test = check_create_dir(get_problem_dir(problem)+'{}/predict/{}_{}/{}/'.format(model.name,problem.config_set_fit['nparam'],problem.config_set_predict['nparam'], 2))
#     supportdir_test = check_create_dir(rootdir_test+'/support_knn15_IWD/')
    
    

#     n_tests = len(query_test)
#     num_predicts = 100
#     #sample_ids = random.sample(np.arange(n_tests).tolist(), num_predicts)
#     sample_ids = [id for id in range(num_predicts)]
#     #sample_ids=[14,63,80,82]
#     #sample_ids=[82]
#     for i in range(len(sample_ids)):
#         si = sample_ids[i]
#         sub_supportdir_test = check_create_dir(supportdir_test+'/'+str(si)+'/')
#         # Find nearest neighbors on snapshots space
#         snapshot_distances, snapshot_index_neighbors = get_KNN_snapshots(target_basis_train,target_test[si],n_neighbors=15)
#         min_val = torch.min(snapshot_distances)
#         idx = torch.argmin(snapshot_distances)
#         if min_val <= 1.e-8:
#             print('close point')
#             alpha = torch.zeros(snapshot_distances.shape[0],device=device,dtype=torch.float64)
#             alpha[idx] = 1.0
#         else:
#             alpha = 1./snapshot_distances**1
#             alpha = alpha/torch.sum(alpha)
#             alpha= alpha.to(dtype=torch.float64)
        

#         # KNN
#         # IS= [0,1,4,5,11,12]
#         # select_indexes = [snapshot_index_neighbors.flatten().tolist()[id] for id in IS]
#         # myfields = [snapshots_basis_train[id] for id in select_indexes]

#         myfields = [snapshots_basis_train[id] for id in snapshot_index_neighbors.flatten().tolist()]
#         #myfields.append(target_test[si].field[0])
#         mymeasures = torch.cat([field[None,None,:,:] for field in myfields], dim=1)
#         weights = alpha[None,:]

#         # uniform weights
#         # myweights = torch.ones((1,snapshot_distances.shape[0]), device=device)/snapshot_distances.shape[0]
#         # random weights
#         # myweights = torch.rand((1,snapshot_distances.shape[0]),device=device)
#         # myweights.data = mat2simplex(myweights)
        

#         bary = ImagesBarycenter_v2(measures=mymeasures,weights=weights, **model.params_sinkhorn_bary)

#         # Nearest Neights
        
#         # Nearest = snapshots_basis_train[snapshot_index_neighbors[0]]
#         # bary = Nearest[None,None,:,:]

#         # GSSP 
#         # bary, weights, evolution = projection_best_ImagesBarycenter(target_field=target_test[si].field[0][None,None,:,:],
#         #                                                             measures= mymeasures,
#         #                                                             field_coordinates= field_coordinates_test,
#         #                                                             params_opt=model.params_opt_best_barycenter,
#         #                                                             params_sinkhorn_bary =model.params_sinkhorn_bary
#         #                                                             )

#         err = ImagesLoss(target_test[si].field[0][None,None,:,:], bary,blur=0.0001,scaling=0.9).item()
#         print('err',err)
        
#         errors.append(err)
#         predictions.append(bary)

#         ## snapshot
#         my_fig_title = 'Approximation ' + str(sample_ids[i])
#         fig_opts={'colors': [], 'labels': None, 'titles': ['ref', 'fit'], 'type': 'plt', 'rootdir': sub_supportdir_test,  'fig_title': my_fig_title, 'format': '.pdf'}
#         plot_fields_images(fields= [target_test.field[sample_ids[i]].detach().cpu().numpy(), bary[0,0].detach().cpu().numpy()],
#                         spatial_coordinates= field_coordinates_test.cpu().numpy(),fig_opts=fig_opts,)
#         ## save neighbors
#         indexes = snapshot_index_neighbors.flatten().cpu().numpy()
#         #indexes = np.array(select_indexes)

#         features = [field.detach().cpu().numpy() for field in myfields]
#         plot_images(indexes,features,spatial_coordinates=field_coordinates_test.cpu().numpy(),fig_opts=fig_opts)
#         my_fig_title = 'ref_bary ' + str(sample_ids[i])
#         fig_opts={'colors': [], 'labels': None, 'titles': ['ref', 'fit'], 'type': 'plt', 'rootdir': sub_supportdir_test,  'fig_title': my_fig_title, 'format': '.pdf'}
#         plot_images_seperate([target_test.field[sample_ids[i]].detach().cpu().numpy(), bary[0,0].detach().cpu().numpy()],spatial_coordinates=field_coordinates_test.cpu().numpy(),fig_opts=fig_opts)
        
#         # save evolution loss
#         # fig, ax= plt.subplots()
#         # im1 = ax.plot(1 + np.arange(len(evolution['loss'])),evolution['loss'],color='r', linestyle='-', marker='o', label='L')
#         # ax.set_title("Loss")
#         # ax.set_xlabel('Iteration')
#         # ax.set_yscale('log')
#         # fig.tight_layout()
#         # fig.savefig(sub_supportdir_test+'evolution_loss.pdf')


#         # Save weights
#         fig, ax= plt.subplots()
#         cp=ax.pcolormesh(weights.detach().cpu().numpy().reshape(-1,5),cmap='jet',edgecolors='w', linewidths=2)
#         fig.colorbar(cp)
#         fig.tight_layout()
#         fig.savefig(sub_supportdir_test+'weight_IWD.pdf')

#     print('errors',errors)
#     # save errors
#     np.save(supportdir_test+'fit_error',np.array(errors),allow_pickle=True)
#     #np.save(supportdir_test+'ref_error',np.array(losses_test[sample_ids]),allow_pickle=True)

#     fig = plt.figure()
#     plt.plot(np.arange(len(errors)),errors,color='b', linestyle='-',marker= '*', linewidth=2)
#     plt.legend(['fit'])
#     plt.yscale('log')
#     plt.savefig(supportdir_test + 'loss.pdf')
#     plt.close()
    
def debug_visualization(x,U):
#Used for debugging
    fig = plt.figure()
    for i in range(U.shape[1]):
        plt.plot(x.numpy(),U[0][i][:].detach().squeeze().numpy())
        plt.show()
    #plt.pause(0.1)
    # plt.close()


def plots_fit(problem, models):
    """For a given problem, analyze convergence of fitting routines of each model
    """
    field_coordinates, _, _, _ = problem.load_dataset(**problem.config_set_fit)
    colors = get_colors(len(models))
    fig, ax = plt.subplots(1, sharex=True, sharey=True)
    fig.suptitle('Fitting: Convergence Error')

    for i, model in enumerate(models):
        # Load fit
        dir_fit = get_fit_dir(problem, model)
        model.load_fit(dir_fit)
        dims = 2+np.arange(len(model.fit_conv))
        # dims = np.arange(len(model.fit_conv))
        ax.plot(dims, model.fit_conv, color=colors[i], linestyle='-', marker='o', label=model.name)
        # ax[1].plot(dims, model.fit_av_err, color=colors[i], linestyle='-', marker='*', label=model.name)
        
    #DEBUG:
    # print('Shape of final bary', model.final_barycenter.shape)
    # print('Shape of field coord', field_coordinates.shape)
    # debug_visualization(field_coordinates, model.final_barycenter)
    
    ax.legend(loc='best')
    # ax[1].legend(loc='best')
    ax.set_xlabel('Dimension n')
    # ax[1].set_xlabel('Dimension n')
    ax.set_ylabel('Max Error')
    # ax[1].set_ylabel('Av. Error')
    plt.yscale('log')
    rootdir = get_problem_dir(problem)
    fig.savefig(rootdir+'fit_conv.pdf')
    
    print('model.fit_conv', model.fit_conv)

def plots_predict(problem, models):
    """Analyse prediction error of each model
    """
    colors = get_colors(len(models))
    fig, ax = plt.subplots(1, sharex=True, sharey=True)
    fig.suptitle('Prediction: Error')

    for i, model in enumerate(models):
        dir_predict = get_predict_dir(problem, model)
        err = pickle.load(open(dir_predict+'err_predict', "rb" ))
        err_max = [max([er**2 for er in errn]) for errn in err]
        err_mean = [mean([er**2 for er in errn]) for errn in err]
        dims = 2+np.arange(len(err))
        ax.plot(dims, err_max, color=colors[i], linestyle='-', marker='o', label='max')
        ax.plot(dims, err_mean, color=colors[i], linestyle='--', marker='*',label='mean')
    ax.legend(loc='best')
    ax.set_xlabel('Dimension n')
    ax.set_ylabel('Error')
    plt.yscale('log')
    rootdir = get_problem_dir(problem)
    fig.savefig(rootdir+'predict_conv.pdf')
# =============================================================================================================
def plots_predict_images(problems, model):
    """Analyse prediction error of each model
    """
    colors = get_colors(len(problems))
    fig, ax = plt.subplots(1, sharex=True, sharey=True)
    fig.suptitle('Prediction: Error')
    err_max = []
    err_mean = []
    dims = []
    for i, problem in enumerate(problems):
        dir_predict = get_predict_dir_v2(problem, model)
        ext = '_{}_{}'.format(problem.config_set_fit['nparam'], problem.config_set_predict['nparam'])
        err = pickle.load(open(dir_predict+'err_predict'+ext, "rb" ))
        err_max.append(max(err)) 
        err_mean.append(mean(err))
        dims.append(problem.config_set_fit['nparam'])


    ax.plot(dims, err_max, color=colors[0], linestyle='-', marker='o', label='Max_errors')
    ax.plot(dims, err_mean, color=colors[1], linestyle='--', marker='o',label='Mean_errors')
    ax.legend(loc='best')
    ax.set_xlabel('snapshots')
    ax.set_ylabel('Error')
    rootdir = get_problem_dir(problems[0])
    fig.savefig(rootdir+'compare_error.pdf')

def plots_predict_knn(problems, model):
    """Analyse prediction error of each model
    """
    colors = get_colors(len(problems))
    fig, ax = plt.subplots(1, sharex=True, sharey=True)
    fig.suptitle('Prediction: Error')
    err_max = []
    err_mean = []
    dims = []
    errors = []
    for i, problem in enumerate(problems):
        dir_predict = get_predict_dir_v2(problem, model)
        ext = '_{}_{}'.format(problem.config_set_fit['nparam'], problem.config_set_predict['nparam'])
        err = pickle.load(open(dir_predict+'err_predict'+ext, "rb" ))
        err_max.append(max(err)) 
        err_mean.append(mean(err))
        errors.append(err)
        dims.append(problem.config_set_fit['nparam'])


    ax.plot(dims, err_max, color=colors[0], linestyle='-', marker='o', label='Max_errors')
    ax.plot(dims, err_mean, color=colors[1], linestyle='--', marker='o',label='Mean_errors')
    ax.legend(loc='best')
    ax.set_xlabel('snapshots')
    ax.set_ylabel('Error')
    rootdir = get_problem_dir(problems[0])
    fig.savefig(rootdir+'compare_error.pdf')
    

    n_tests = len(errors[0])
    num_predicts = 5
    sample_ids = random.sample(np.arange(n_tests).tolist(), num_predicts)
    colors = get_colors(n_tests)
    fig, ax = plt.subplots(1, sharex=True, sharey=True)
    fig.suptitle('Prediction: Error')
    for id in sample_ids:
        error_id = [errors[p][id] for p in range(len(problems))]
        ax.plot(dims, error_id, color=colors[id], linestyle='-', marker='o', label='Index_'+str(id))
    ax.legend(loc='best')
    ax.set_xlabel('snapshots')
    ax.set_ylabel('Error')
    rootdir = get_problem_dir(problems[0])
    fig.savefig(rootdir+'sample_error.pdf')
    



    
# def plots_predict_images_times(problems, model):
#     """Analyse prediction error of each model
#     """
#     colors = get_colors(len(problems))
#     fig, ax = plt.subplots(1, sharex=True, sharey=True)
#     fig.suptitle('Prediction: Time')
#     time_max = []
#     time_mean = []
#     dims = []
#     for i, problem in enumerate(problems):
#         dir_predict = get_predict_dir_v2(problem, model)
#         prediction_times = pickle.load(open(dir_predict+'prediction_times', "rb" ))
#         time_max.append(max(prediction_times))
#         time_mean.append(mean(prediction_times))
#         dims.append(problem.name.split('_')[1])


#     ax.plot(dims, time_max, color=colors[0], linestyle='-', marker='o', label='Max_times')
#     ax.plot(dims, time_mean, color=colors[1], linestyle='--', marker='o',label='Mean_times')
#     ax.legend(loc='best')
#     ax.set_xlabel('M')
#     ax.set_ylabel('Times (s) ')
#     rootdir = get_problem_dir(problems[0])
#     fig.savefig(rootdir+'compare_error.pdf')


def impact_training_samples():
    """Analyse role of number of training samples in fit and predict
    """
    pass


