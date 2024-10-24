# SPDX-FileCopyrightText: The sinkhorn-rom authors, see file AUTHORS
#
# SPDX-License-Identifier: GPL-3.0-or-later

import torch
import os
# os.environ['CUDA_PATH'] = '/home/prai/anaconda3/lib/python3.10/site-packages/triton/third_party/cuda/include/cuda.h'
import uuid

from ...config import data_dir, device
from ...utils import pprint

def get_file_path(filename):
  dataset_file = data_dir+filename
  if os.path.exists(dataset_file):
    print('Loading '+filename)
    return dataset_file
  else:
    raise Exception('File '+dataset_file+' not found.')

class Problem():
    def __init__(self, name='Gaussian1d',
                       id = 'default',
                       config_set_fit={'nparam': 500, 'id_set': 0},
                       config_set_predict={'nparam': 500, 'id_set': 1}):
        pprint('Problem ='+ name, font_style='bold', fg='White', bg='Green')
        self.name = name
        self.id = id
        self.config_set_fit = config_set_fit
        self.config_set_predict = config_set_predict

    def load_dataset(self, nparam=500, id_set=0):
        return self.load_points(nparam=nparam, id_set=id_set), \
               self.load_fields(nparam=nparam, id_set=id_set), \
               self.load_params(nparam=nparam, id_set=id_set), \
               self.load_uuid(nparam=nparam, id_set=id_set)

    def load_points(self, nparam=500, id_set=0):
        return torch.load(get_file_path(self.name+'/{}/{}/points'.format(nparam, id_set)), map_location=device)

    def load_params(self, nparam=500, id_set=0):
        return torch.load(get_file_path(self.name+'/{}/{}/params'.format(nparam, id_set)), map_location=device)

    def load_fields(self, nparam=500, id_set=0):
        return torch.load(get_file_path(self.name+'/{}/{}/fields'.format(nparam, id_set)), map_location=device)

    def load_uuid(self, nparam=500, id_set=0):
        with open(get_file_path(self.name+'/{}/{}/uuid.txt'.format(nparam, id_set)), "r") as uuid_file: # load unique identifier
            return uuid_file.read()

class Gaussian1d(Problem):
    def __init__(self):
        super().__init__()

class Gaussian2d(Problem):
    def __init__(self):
        super().__init__()

class VlasovPoisson(Problem):
    def __init__(self):
        super().__init__()

class Rectangle2d(Problem):
    def __init__(self):
        super().__init__()
        
class Transport1d(Problem):
    def __init__(self):
        super().__init__()
class Burger2d(Problem):
    def __init__(self):
        super().__init__()
        
class Burger1d(Problem):
    def __init__(self):
        super().__init__()
        
class KdV1d(Problem):
    def __init__(self):
        super().__init__()

class ViscousBurger1d(Problem):
    def __init__(self):
        super().__init__()
        
class CamassaHolm1d(Problem):
    def __init__(self):
        super().__init__()

class Heavyside1d(Problem):
    def __init__(self):
        super().__init__()
        
class Shallow_Water2d(Problem):
    def __init__(self):
        super().__init__()