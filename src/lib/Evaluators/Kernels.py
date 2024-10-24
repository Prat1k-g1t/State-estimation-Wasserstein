# SPDX-FileCopyrightText: The sinkhorn-rom authors, see file AUTHORS
#
# SPDX-License-Identifier: GPL-3.0-or-later

import torch
import numpy as np
from scipy.spatial.distance import cdist
from itertools import product
import time
from scipy.sparse import csr_matrix

from .Metrics import Metrics

class KernelFunctions:
    @staticmethod
    def gaussian_kernel(x, y, sigma, metric='euclidean'):
        """

        @param x: vector 1
        @param y: vector 2
        @param sigma: sigma of the gaussian kernel
        @return:
        """
        # the if condition with the axis is to account for many vectors simultaneously, first dimension is the iterator,
        # the others correspond to the vector.

        return np.exp(-Metrics.dist(x,y, metric=metric)** 2 / sigma ** 2)

    @staticmethod
    def exponential_kernel(x, y, sigma, metric='euclidean'):
        """

        @param x: vector 1
        @param y: vector 2
        @param sigma: sigma of the gaussian kernel
        @return:
        """
        # the if condition with the axis is to account for many vectors simultaneously, first dimension is the iterator,
        # the others correspond to the vector.

        return np.exp(-Metrics.dist(x,y, metric=metric)** 2 / sigma)