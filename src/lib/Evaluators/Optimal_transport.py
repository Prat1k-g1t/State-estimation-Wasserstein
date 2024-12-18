# SPDX-FileCopyrightText: The sinkhorn-rom authors, see file AUTHORS
#
# SPDX-License-Identifier: GPL-3.0-or-later

import numpy as np
from scipy.sparse.linalg import spsolve

from pysdot import PowerDiagram

from scipy.sparse import csr_matrix

from pysdot.domain_types import ConvexPolyhedraAssembly
from pysdot import PowerDiagram


# utilities
def make_square(box=[0, 0, 1, 1]):
    """
    Constructs a square domain with uniform measure (source measure 'rho').
    To be passed to the 'newton_ot' and 'PowerDiagram' functions.
    Args:
        box (list): coordinates of the bottom-left and top-right corners
    Returns:
        domain (pysdot.domain_types): domain
    """
    domain = ConvexPolyhedraAssembly()
    domain.add_box([box[0], box[1]], [box[2], box[3]])
    return domain

def laguerre_areas(domain, Y, psi, der=False):
    """
    Computes the areas of the Laguerre cells intersected with the domain.
    Args:
        domain (pysdot.domain_types): domain of the (continuous)
                                      source measure
        Y (np.array): points of the (discrete) target measure
        psi (np.array or list): Kantorovich potentials
        der (bool): wether or not return the Jacobian of the areas
                    w.r.t. psi
    Returns:
        pd.integrals() (list): list of areas of Laguerre cells
    """
    pd = PowerDiagram(Y, -psi, domain)
    if der:
        N = len(psi)
        mvs = pd.der_integrals_wrt_weights()
        return mvs.v_values, csr_matrix((-mvs.m_values, mvs.m_columns, mvs.m_offsets), shape=(N, N))
    else:
        return pd.integrals()




# Newton's algorithm
def newton_ot(domain, Y, nu, psi0=None, verbose=False,maxerr=1e-6, maxiter=20, t_init=1.):
    """
    Computes the Kantorovich potentials associated to the
    transport of 'domain' to 'Y' with masses 'nu'.
    Args:
        domain (pysdot.domain_types): domain of the (continuous)
                                      source measure
        Y (np.array): support points of the (discrete) target measure
        nu (np.array or list): weights (measure) of points Y
        psi0 (np.array or list): initialization of Kantorovich potentials
        verbose (bool): wether or not to print the norm of the error along
                        iterations (error is the difference between laguerre
                        cell areas and weights of nu)
        maxerr (float): threshold error under which the algorithm stops
        maxiter (int): threshold iteration under which the algorithm stops
    Returns:
        psi (np.array): list of Kantorovich potentials
    """
    if psi0 is None:
        psi0 = np.zeros(len(nu))
        
    def F(psip):
        g, h = laguerre_areas(domain, Y, np.hstack((psip,0)), der=True)
        return g[0:-1], h[0:-1,0:-1]
    
    psip = psi0[0:-1] - psi0[-1]
    nup = nu[0:-1]
    g, h = F(psip)
    for it in range(maxiter):
        err = np.linalg.norm(nup - g)
        if verbose:
            print("it %d: |err| = %g" % (it, err))
        if err <= maxerr:
            break
        d = spsolve(h, nup - g)
        t = t_init 
        psip0 = psip.copy()
        while True:
            psip = psip0 + t * d
            g,h = F(psip)
            if np.min(g) > 0:
                break
            else:
                t = t/2
    return np.hstack((psip,0))


# class for semi-discrete OT
class semi_discrete_ot:
    """
    Class for computing a set of transport plans between the 
    Lebesgue measure on [0, 1]² and a set of point clouds.
    """

    def __init__(self, grid_size, *args, **kwargs):
        """
        Args:
            grid_size (int): discretization parameter for the transport plans
                             (m in the paper)
        """
        # source domain
        self.domain = make_square()
        # discretize source domain (unit square)
        self.grid_size = grid_size 
        
    def fit_transport_plans(self, point_clouds, masses=None, rescaling=True,
                            maxerr=1e-6, maxiter=20, t_init=1.,
                            *args, **kwargs):
        """
        Fits transport plans to point clouds.
        Args:
            point_clouds (dict): dictionary of point clouds
            masses (dict): dictionary of masses assigned to each point cloud
                           (default will be uniform masses for each point cloud)
            rescaling (bool): rescale or not the coordinates of point clouds
                              to fit in [0, 1]² to make computations easier 
                              (rescaling undone when returning transport plans)
            maxerr (float): threshold error under which Newton's algo stops
            maxiter (int): threshold iteration under which Newton's algo stops
            t_init (float): inital value of t for Newton's algorithm 
        """
        nb_clouds = len(point_clouds)
        cloud_keys = list(point_clouds.keys())
        # compute the transport plan of each cloud
        self.potentials = {}
        self.transport_plans = np.zeros((nb_clouds, 2 * self.grid_size**2))
        for c in range(nb_clouds):
            # get point cloud
            sample = point_clouds[cloud_keys[c]].astype('float64')
            N = len(sample)
            if rescaling:
                # rescale to [0, 1]²
                sample_min, sample_max = np.min(sample, axis=0), np.max(sample, axis=0)
                sample = (sample - sample_min) / (sample_max - sample_min) 
            # build target measure
            if masses is not None:
                nu = masses[cloud_keys[c]]
            else:
                nu = np.ones(N)/N
            # compute OT between source and target (get Kantorovich potentials)
            self.potentials[c] = newton_ot(self.domain, sample, nu, psi0=None, verbose=False,maxerr=maxerr, maxiter=maxiter, t_init=t_init)
            # build the *discretized* transport plan associated to these potentials
            pd = PowerDiagram(sample, -self.potentials[c], self.domain)
            img = pd.image_integrals([0, 0], [1, 1], [self.grid_size, self.grid_size])
            img /= np.expand_dims(img[ :, :, 2], -1)
            if rescaling:
                # undo rescaling
                img[:, :, :2] = (sample_max - sample_min) * img[:, :, :2] + sample_min 
            # save transport plan            
            self.transport_plans[c] = img[:, :, :2].flatten()