# SPDX-FileCopyrightText: The sinkhorn-rom authors, see file AUTHORS
#
# SPDX-License-Identifier: GPL-3.0-or-later

import matplotlib 
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib import ticker
from matplotlib.colors import Normalize
#matplotlib.use('Agg')
import numpy as np
from typing import List
from scipy.spatial import Voronoi, voronoi_plot_2d

from .config import results_dir, dtype, device
from .utils import check_create_dir
from .lib.Evaluators.Barycenter import ImagesLoss, ImagesBarycenter_v2, mat2simplex
from .tools import barycentric_coordinates

import pandas as pd
import torch
from shapely.geometry import Point, Polygon

def plot_fields(fields: List[np.array], spatial_coordinates: np.array,
                fig_opts={'colors': [], 'labels': None, 'titles': None, 'plot_type': 'voronoi', 'rootdir': results_dir,  'fig_title': 'some_field', 'format': '.pdf'},
                ):
    d = spatial_coordinates.shape[1]
    fig, ax = plt.subplots(len(fields), sharex=True, sharey=True)
    ax = ax if isinstance(ax, np.ndarray) else np.array([ax])
    fig.suptitle('')

    if d==1:        
        # Colors
        if len(fig_opts['colors']) != len(fields):
            colors = get_colors(len(fields))

        for i, field in enumerate(fields):
            #DEBUG:
            # print('field shape',field.transpose(-1,0).shape)
            # print('field.T shape',torch.transpose(field, 0, 1).shape)
            # print('spatial_coordinates',spatial_coordinates.shape)
            if fig_opts['labels'] is None:
                label = ''
            else:
                label = fig_opts['labels'][i]
            
            ax[i].plot(spatial_coordinates, field.transpose(-1,0), color=colors[i], label=label)
            #field.transpose(-1,0)
            if fig_opts['titles'] is None:
                title = ''
            else:
                title = fig_opts['titles'][i]

            ax[i].set_title(title)
            ax[i].legend(loc='best')
    elif d==2:

        if fig_opts['plot_type']=='voronoi':
            
            # find min/max values for normalization
            minima = min([min(field) for field in fields]) 
            maxima = max([max(field) for field in fields]) 
            # normalize chosen colormap
            norm = matplotlib.colors.Normalize(vmin=minima, vmax=maxima, clip=True)
            mapper = cm.ScalarMappable(norm=norm, cmap=cm.hot)
            
            vor = Voronoi(spatial_coordinates)
            for i, field in enumerate(fields):
                # Voronoi tesselation and coloring according to field values
                voronoi_plot_2d(vor, ax = ax[i], show_points=False, show_vertices=False)
                for r in range(len(vor.point_region)):
                    region = vor.regions[vor.point_region[r]]
                    if not -1 in region:
                        # We do not color cells at the boundary
                        polygon = [vor.vertices[i] for i in region]
                        ax[i].fill(*zip(*polygon), color=mapper.to_rgba(field[r]))
            
        elif fig_opts['plot_type']=='scatter':
            for i, field in enumerate(fields):
                if fig_opts['labels'] is None:
                    label = ''
                else:
                    label = fig_opts['labels'][i]
                ax[i].scatter(x=spatial_coordinates[:,0], y=spatial_coordinates[:,1], c=field)
                if fig_opts['titles'] is None:
                    title = ''
                else:
                    title = fig_opts['titles'][i]
                ax[i].set_title(title)
                ax[i].legend(loc='best')
        else:
            print('plot_type {} not implemented.'.format(fig_opts['plot_type']))
    else:
        print('Visualization of fields of dim>=3 not implemented')
        return
    fig_opts['rootdir'] = check_create_dir(fig_opts['rootdir'])
    fig.savefig(fig_opts['rootdir']+fig_opts['fig_title']+fig_opts['format'])

def get_colors(n):
    """Get set of n colors for visualization
    """
    cmap = plt.get_cmap('jet')
    return cmap(np.linspace(0, 1.0, n))

def plot_fields_images( fields: List[np.array], spatial_coordinates: np.array,
                    fig_opts={'colors': [], 'labels': None, 'titles': None, 'plot_type': None, 'rootdir': results_dir,  'fig_title': 'some_field', 'format': '.pdf'},
                    ):
    # print('cor',set(spatial_coordinates[:, 0]))
    if len(fields) >1 :
        fig, ax = plt.subplots(len(fields), sharex=True, sharey=True)
    else :
        fig, ax = plt.subplots(1,1)
    xmin = spatial_coordinates[0, 0] ; xmax = spatial_coordinates[-1, 0]
    ymin = spatial_coordinates[0, 1] ; ymax = spatial_coordinates[-1, 1]
    nx = int( np.sqrt(spatial_coordinates.shape[0]))
    ny = nx
    X = np.linspace(xmin, xmax, num=nx)
    Y = np.linspace(ymin, ymax, num=ny)
    xv, yv = np.meshgrid(X, Y)

    # find min/max values for normalization
    minima = min([np.min(field) for field in fields])
    maxima = max([np.max(field) for field in fields])

    # normalize chosen colormap
    norm = matplotlib.colors.Normalize(vmin=minima, vmax=maxima, clip=True)
    mapper = cm.ScalarMappable(norm=norm, cmap=cm.viridis)
    if fig_opts['titles'] is None:
        fig_opts['titles'] = ['ref','fit']

    if len(fields) >1 :
        for i, field in enumerate(fields):
            cp=ax[i].contourf(xv, yv, field, cmap='jet', vmin=minima, vmax = maxima)
            #plt.colorbar(cp, ax=ax[i])
            ax[i].title.set_text(fig_opts['titles'][i])
            ax[i].set(adjustable='box', aspect='equal')
        fig.colorbar(cp, ax=ax.ravel().tolist())
    else:
        cp=ax.contourf(xv, yv, fields[0], cmap='jet', vmin=minima, vmax = maxima)
        fig.colorbar(cp)
    
    fig_opts['rootdir'] = check_create_dir(fig_opts['rootdir'])
    fig.savefig(fig_opts['rootdir'] + fig_opts['fig_title'] + fig_opts['format'])
    plt.close()
    

def plot_fields_images_nb( fields: List[np.array], spatial_coordinates: np.array,
                    fig_opts={'colors': [], 'labels': None, 'titles': None, 'plot_type': None, 'rootdir': results_dir,  'fig_title': 'some_field', 'format': '.pdf'},
                    ):
    # print('cor',set(spatial_coordinates[:, 0]))
    if len(fields) >1 :
        fig, ax = plt.subplots(len(fields), sharex=True, sharey=True)
    else :
        fig, ax = plt.subplots(1,1)
    xmin = spatial_coordinates[0, 0] ; xmax = spatial_coordinates[-1, 0]
    ymin = spatial_coordinates[0, 1] ; ymax = spatial_coordinates[-1, 1]
    nx = int( np.sqrt(spatial_coordinates.shape[0]))
    ny = nx
    X = np.linspace(xmin, xmax, num=nx)
    Y = np.linspace(ymin, ymax, num=ny)
    xv, yv = np.meshgrid(X, Y)

    # find min/max values for normalization
    minima = min([np.min(field) for field in fields])
    maxima = max([np.max(field) for field in fields])

    # normalize chosen colormap
    norm = matplotlib.colors.Normalize(vmin=minima, vmax=maxima, clip=True)
    mapper = cm.ScalarMappable(norm=norm, cmap=cm.viridis)
    if fig_opts['titles'] is None:
        fig_opts['titles'] = ['ref','fit']

    if len(fields) >1 :
        for i, field in enumerate(fields):
            cp=ax[i].contourf(xv, yv, field, cmap='jet', vmin=minima, vmax = maxima)
            #plt.colorbar(cp, ax=ax[i])
            ax[i].title.set_text(fig_opts['titles'][i])
            ax[i].set(adjustable='box', aspect='equal')
        fig.colorbar(cp, ax=ax.ravel().tolist())
    else:
        cp=ax.contourf(xv, yv, fields[0], cmap='jet', vmin=minima, vmax = maxima)
        fig.colorbar(cp)
    
    fig_opts['rootdir'] = check_create_dir(fig_opts['rootdir'])
    fig.savefig(fig_opts['rootdir'] + fig_opts['fig_title'] + fig_opts['format'])
    plt.show()

def plot_fields_images_v2( fields: List[np.array], spatial_coordinates: np.array,
                    fig_opts={'colors': [], 'labels': None, 'titles': None, 'plot_type': 'voronoi', 'rootdir': results_dir,  'fig_title': 'some_field', 'format': '.pdf'},
                    ):
    # print('cor',set(spatial_coordinates[:, 0]))
    if len(fields) >1 :
        #fig, axes = plt.subplots(len(fields), sharex=True, sharey=True)
        fig, axes = plt.subplots(nrows=1,ncols=2, sharex=True, sharey=True,subplot_kw=dict(box_aspect=1),constrained_layout=True)
    else :
        fig, ax = plt.subplots(1,1)

    xmin = spatial_coordinates[0, 0] ; xmax = spatial_coordinates[-1, 0]
    ymin = spatial_coordinates[0, 1] ; ymax = spatial_coordinates[-1, 1]
    nx = int( np.sqrt(spatial_coordinates.shape[0]))
    ny = nx
    X = np.linspace(xmin, xmax, num=nx)
    Y = np.linspace(ymin, ymax, num=ny)
    xv, yv = np.meshgrid(X, Y)

    # find min/max values for normalization
    minima = min([np.min(field) for field in fields])
    maxima = max([np.max(field) for field in fields])

    # normalize chosen colormap
    norm = matplotlib.colors.Normalize(vmin=minima, vmax=maxima, clip=True)
    cmap=cm.get_cmap('viridis')
    mapper = cm.ScalarMappable(norm=norm, cmap=cmap)

    if fig_opts['titles'] is None:
        fig_opts['titles'] = ['ref','fit']

    if len(fields) >1 :
        for i,ax in enumerate(axes.flat):
            ax.contourf(xv, yv, fields[i], cmap=cmap, norm=norm)
            ax.title.set_text(fig_opts['titles'][i])
            #ax.set(adjustable='box', aspect='equal')
        #fig.colorbar(cp, ax=ax.ravel().tolist())
        fig.colorbar(mapper, ax=axes.flat,shrink=0.6)
        
    else:
        cp=ax.contourf(xv, yv, fields[0], cmap=cmap, norm=norm)
        fig.colorbar(cp)
    #plt.tight_layout()
    fig_opts['rootdir'] = check_create_dir(fig_opts['rootdir'])
    fig.savefig(fig_opts['rootdir'] + fig_opts['fig_title'] + fig_opts['format'])
    plt.close()



def plot_batch_images(indexes,features,spatial_coordinates: np.array,
                    fig_opts={'colors': [], 'labels': None, 'titles': None, 'rootdir': results_dir,  'fig_title': 'some_field', 'format': '.pdf'}
                        ):

    number_images = indexes.shape[0]
    fig, axes = plt.subplots(number_images, sharex=True, sharey=True)
    
    xmin = spatial_coordinates[0, 0] ; xmax = spatial_coordinates[-1, 0]
    ymin = spatial_coordinates[0, 1] ; ymax = spatial_coordinates[-1, 1]
    nx = int( np.sqrt(spatial_coordinates.shape[0]))
    ny = nx
    X = np.linspace(xmin, xmax, num=nx)
    Y = np.linspace(ymin, ymax, num=ny)
    xv, yv = np.meshgrid(X, Y)

    # find min/max values for normalization
    minima = np.min(features)
    maxima = np.max(features)
    print('min',minima)
    print('max',maxima)
    # normalize chosen colormap
    #norm = matplotlib.colors.Normalize(vmin=minima, vmax=maxima, clip=True)
    #mapper = cm.ScalarMappable(norm=norm, cmap=cm.viridis)
    title = [str(indexes[i]) for i in range(number_images)]
    
    for i,ax in enumerate(axes.flat):
        im=ax.contourf(xv, yv, features[i], vmin=minima, vmax = maxima,cmap="jet")
        ax.title.set_text(title[i])
        im.set_clim(minima,maxima)
    
    fig.colorbar(im, ax=axes.ravel().tolist())

    
    fig_opts['rootdir'] = check_create_dir(fig_opts['rootdir'])
    fig.savefig(fig_opts['rootdir'] + fig_opts['fig_title'] + fig_opts['format'])

# def plot_images(indexes,features,spatial_coordinates: np.array,
#                     fig_opts={'colors': [], 'labels': None, 'titles': None,'type': 'plt', 'rootdir': results_dir,  'fig_title': 'some_field', 'format': '.pdf'}
#                         ):

#     number_images = indexes.shape[0]
#     xmin = spatial_coordinates[0, 0] ; xmax = spatial_coordinates[-1, 0]
#     ymin = spatial_coordinates[0, 1] ; ymax = spatial_coordinates[-1, 1]
#     nx = int( np.sqrt(spatial_coordinates.shape[0]))
#     ny = nx
#     X = np.linspace(xmin, xmax, num=nx)
#     Y = np.linspace(ymin, ymax, num=ny)
#     #xv, yv = np.meshgrid(X, Y,indexing='ij')
#     xv, yv = np.meshgrid(X, Y)

#     minima = min([np.min(field) for field in features])
#     maxima = max([np.max(field) for field in features])



#     title = [str(indexes[i]) for i in range(number_images)]
#     fig_opts['rootdir'] = check_create_dir(fig_opts['rootdir'])
#     for i in range(number_images):
#         if fig_opts['type']=='plt':
#             fig, ax= plt.subplots()
#             im=ax.contourf(xv, yv, features[i],vmin=minima, vmax = maxima,cmap="jet")
#             ax.title.set_text(title[i])
#             fig.colorbar(im)
#             fig_name = fig_opts['rootdir'] + fig_opts['fig_title']+'_i_'+str(i)+'_Id_'+str(indexes[i]) + fig_opts['format']
#             fig.savefig(fig_name)
#             plt.close()
#         else:
#             fig = mlab.figure()
#             fig_name = fig_opts['rootdir'] + fig_opts['fig_title']+'_i_'+str(i)+'_Id_'+str(indexes[i]) + fig_opts['format']
#             mlab.imshow(features[i])
#             mlab.colorbar(orientation='vertical')
#             mlab.view(0,0)
#             mlab.savefig(fig_name)
#             #mlab.show()
#             #mlab.close(fig)


def plot_sample_images(indexes, fields,fig_opts={'colors': [], 'titles': None, 'rootdir': None,  'fig_title': 'some_field', 'format': '.png'}):
    # fields : list
    number_images = indexes.shape[0]
    fig, axes = plt.subplots(nrows=3,ncols=3, sharex=True, sharey=True,subplot_kw=dict(box_aspect=1),constrained_layout=True)
    
    xmin = 0
    xmax = 10
    ymin = 0
    ymax = 10

    ny = nx =64
    X = np.linspace(xmin, xmax, num=nx)
    Y = np.linspace(ymin, ymax, num=ny)
    xv, yv = np.meshgrid(X, Y)

    # find min/max values for normalization
    minima = min([np.min(field) for field in fields])
    maxima = max([np.max(field) for field in fields])

    # normalize chosen colormap
    norm = Normalize(vmin=minima, vmax=maxima, clip=True)
    cmap=cm.get_cmap('viridis')
    mapper = cm.ScalarMappable(norm=norm, cmap=cmap)
    

    # mapper = cm.ScalarMappable(norm=norm, cmap=cm.jet)
    # cmap=cm.get_cmap('jet')
    
    title = [str(indexes[i]) for i in range(number_images)]
    
    for i,ax in enumerate(axes.flat):
        #cp=ax.contourf(xv, yv, fields[i], cmap='jet', vmin=minima, vmax = maxima)
        #plt.colorbar(cp, ax=ax[i])
        ax.contourf(xv, yv, fields[i],cmap=cmap, norm=norm)
        ax.title.set_text('Id = '+ title[i])
        #ax.set(adjustable='box', aspect='equal')
        #ax.set(aspect='equal')
    #fig.colorbar(cp, ax=axes.ravel().tolist())
    #fig.colorbar(cp, ax=axes.flat)

    fig.colorbar(mapper, ax=axes.flat)
    
    
    #plt.tight_layout()
    fig_opts['rootdir'] = check_create_dir(fig_opts['rootdir'])
    fig.savefig(fig_opts['rootdir'] + fig_opts['fig_title'] + fig_opts['format'])
    plt.close()










def plot_fields_weights( fields: List[np.array],
                    fig_opts={'colors': [], 'labels': None, 'titles': None, 'plot_type': 'voronoi', 'rootdir': results_dir,  'fig_title': 'weights', 'format': '.pdf'},
                    ):
    # print('cor',set(spatial_coordinates[:, 0]))
    fig, ax = plt.subplots(len(fields), sharex=True, sharey=True)
    

    # find min/max values for normalization
    minima = min([np.min(field) for field in fields])
    maxima = max([np.max(field) for field in fields])

    # normalize chosen colormap
    norm = matplotlib.colors.Normalize(vmin=minima, vmax=maxima, clip=True)
    mapper = cm.ScalarMappable(norm=norm, cmap=cm.viridis)

    title = ['ref','fit']
    
    for i, field in enumerate(fields):
        cp=ax[i].pcolormesh(field.reshape(-1,10),cmap='jet', vmin=minima, vmax = maxima,edgecolors='w', linewidths=2)
        #plt.colorbar(cp, ax=ax[i])
        ax[i].title.set_text(title[i])
    fig.colorbar(cp, ax=ax.ravel().tolist())

    fig_opts['rootdir'] = check_create_dir(fig_opts['rootdir'])
    fig.savefig(fig_opts['rootdir'] + fig_opts['fig_title'] + fig_opts['format'])

def plot_evolution(evolution, fig_opts={'rootdir': results_dir, 'format':'.pdf'}):
    # visualization evolution
    root_dir = fig_opts['rootdir']
    idims = 1 + np.arange(len(evolution['loss']))
    len_supp = [supp.shape[0] for supp in evolution['support']]
    print('total_loss',evolution['loss'])
    total_loss = evolution['loss']
    total_error = evolution['true_error']

    df_loss = pd.DataFrame(total_loss,columns =['loss'])
    df_supp = pd.DataFrame(len_supp,columns =['support'])
    df_error = pd.DataFrame(total_error,columns =['error'])
    df_loss.to_csv(root_dir+'loss'+'.csv')
    df_supp.to_csv(root_dir+'support'+'.csv')
    df_error.to_csv(root_dir+'error'+'.csv')

    fig, axs = plt.subplots(1, 3,figsize=(8.5, 8))
    im1 = axs[0].plot(idims,total_loss,color='r', linestyle='-', marker='o', label='L')
    axs[0].set_yscale('log')
    axs[0].set_title("Loss")
    axs[0].set_xlabel('Iteration')

    im2 =axs[1].plot(idims,len_supp,color='b', linestyle='-', marker='o', label='S')
    axs[1].set_title("Support")
    axs[1].set_xlabel('Iteration')

    im3 = axs[2].plot(idims,total_error,color='g', linestyle='-', marker='o', label='E')
    axs[2].set_yscale('log')
    axs[2].set_title("Error")
    axs[2].set_xlabel('Iteration')

    fig.tight_layout()
    fig.savefig(root_dir+'evolution_gamma'+fig_opts['format'])

def plot_Loss_map(main_points, main_weights ,target,measures,type_loss='W2',distance_target_measures=None, point_ref=None,evolution=None,fig_opts={'rootdir':results_dir,'format':'.pdf'}):
    N = measures.shape[1]
    total_measures = torch.cat([measures[0][id][None,None,:,:] for id in range(N)], dim=0)
    targets =  target.repeat(N, 1, 1, 1).to(dtype=dtype)
    if distance_target_measures is None:
        distance_true = ImagesLoss(targets,total_measures,blur=0.001,scaling=0.8)
    else:
        distance_true = distance_target_measures.to(dtype=dtype, device=device)

    X=main_points.cpu().numpy().tolist()
    X= [np.array(x) for x in X]

    polygon = Polygon(X)
    xv = yv = np.linspace(0., 1., 40)
    Xgrid, Ygrid = np.meshgrid(xv, yv)

    Z = np.zeros(Xgrid.shape)

    interior_polygon = np.zeros(Xgrid.shape)
    for index, _ in np.ndenumerate(Xgrid):
        interior_polygon[index] = polygon.covers(Point(Xgrid[index], Ygrid[index])) 
        
        if interior_polygon[index]==1:
            point = np.array([Xgrid[index], Ygrid[index]])
            W= barycentric_coordinates(X, point)
            weight = torch.tensor(W,dtype=dtype,device=device)
            bary = ImagesBarycenter_v2(measures=measures, weights=weight[None,:], blur=0.001, scaling_N = 300,backward_iterations=5)
            if type_loss == 'mse':
                barycenters = bary.repeat(N,1,1,1)
                distance_approx = ImagesLoss(barycenters,total_measures,blur=0.001,scaling=0.8)
                loss = N*torch.nn.MSELoss()(distance_true, distance_approx)
            elif type_loss =='W2':
                loss = ImagesLoss(target, bary,blur=0.0001,scaling=0.9)
            else:
                print('Use W2 for the loss function !')
                loss = ImagesLoss(target, bary,blur=0.0001,scaling=0.9)

            Z[index] = loss.item()
    Z[interior_polygon == 0] = np.ma.masked
    if evolution is not None:
        evolution_weights = torch.vstack(evolution['weight'])
        print(evolution_weights)
        evolution_points = torch.matmul(evolution_weights,main_points)
        evolution_losses = torch.tensor(evolution['loss'])
        print(evolution_losses)
        mypoints= evolution_points.cpu().numpy()
        myloss = evolution_losses.cpu().numpy()
        Loss = np.true_divide(myloss, np.max(myloss))
        # create map color
        indexColor = np.floor(Loss*255)
        cmap = plt.cm.get_cmap('jet')
        colors = cmap(np.arange(cmap.N))

    fig, ax = plt.subplots()
    #nlevels = 8
    #levels = np.logspace(-8, -1, num=nlevels)
    cs = ax.contourf(Xgrid, Ygrid, Z, locator=ticker.LogLocator(), cmap='jet') # locator=ticker.LogLocator()
    if point_ref is not None:
        ax.plot(*point_ref[0].cpu().numpy(), marker="o", markersize=10, markeredgecolor="red", markerfacecolor="red")
    if evolution is not None:
        for i in range(mypoints.shape[0]):
            ax.plot(mypoints[i][0],mypoints[i][1],'o',color=colors[int(indexColor[i])])
    cbar = fig.colorbar(cs)
    fig.savefig(fig_opts['rootdir']+'Loss_map'+ fig_opts['format'])
    #plt.show()


