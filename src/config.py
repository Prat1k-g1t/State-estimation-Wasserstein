# SPDX-FileCopyrightText: The sinkhorn-rom authors, see file AUTHORS
#
# SPDX-License-Identifier: GPL-3.0-or-later

import os
# os.environ['CUDA_PATH'] = '/home/prai/anaconda3/lib/python3.10/site-packages/triton/third_party/cuda/include/cuda.h'
import torch
from typing import List
from sys import platform

# Folders
project_name = 'sinkhorn-rom'
project_dir  = os.path.dirname(os.path.dirname(__file__))
results_dir  = project_dir + '/results/'
data_dir     = project_dir + '/data/'
fit_dir      = project_dir + '/fit/'

# For pytorch: get device and set dtype
use_cuda = torch.cuda.is_available()
#use_cuda = False
# device = torch.device('cuda') if use_cuda else torch.device('cpu')
device = torch.device('cpu')
dtype = torch.double # Autograd does not work so far with torch.double
# dtype = torch.float

# Use Keops only if on Linux or MacOS
#use_pykeops = True if platform.startswith(('linux', 'darwin')) else False
use_pykeops = True
