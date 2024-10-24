# SPDX-FileCopyrightText: The sinkhorn-rom authors, see file AUTHORS
#
# SPDX-License-Identifier: GPL-3.0-or-later

import numbers
from typing import List
import numpy as np
import torch

from ...config import device


class DataStruct:
    def __init__(self, **kwargs):
        lengths = [len(v) if isinstance(v, (list, np.ndarray, torch.Tensor)) else 1 for v in kwargs.values()]
        lengths_not_1 = [l for l in lengths if l != 1]
        if len(lengths_not_1) > 0:
            self.data_size = lengths_not_1[0]
            assert all(
                [l == self.data_size for l in lengths_not_1]), 'all arguments should have length 1 or same length l'
        else:
            self.data_size = 1

        # self.data = kwargs

        for k, v in kwargs.items():
            setattr(self, k, v)
        self.names = list(kwargs.keys())
        self.var_lengths = {k: l for k, l in zip(self.names, lengths)}

    def __len__(self):
        return self.data_size

    def __getitem__(self, i):
        if isinstance(i, (numbers.Integral, str)):
            res = self.__getitem__([i])
        elif isinstance(i, (List, np.array)):
            if isinstance(i[0], str):
                assert len(set(i).intersection(self.names)) == len(i), \
                    'names should be in DataStruct but i:{} and names are: {}'.format(i, self.names)
                res = DataStruct(**{k: getattr(self, k) for k in i})
            elif isinstance(i[0], numbers.Integral):
                res = DataStruct(**{
                    k: [getattr(self, k)[ix] if self.var_lengths[k] == self.__len__() else getattr(self, k)[0] for ix in i]
                    for k in self.names})
            else:
                raise Exception('Type of indexer elements {} not implemented'.format(type(i)))
        else:
            raise Exception('Type of indexer i {} not implemented'.format(type(i)))
        return res

    def __iter__(self):
        for i in range(self.__len__()):
            yield self[[i]]

    def shape(self):
        return tuple([self.__len__()] + [np.shape(getattr(self, k)[0]) for k in self.names])

    def delete_index(self, index_list: List):
        """
            Remove list of elements indexes by index_list from DataStruct
        """
        kwargs = {}
        for key, length in self.var_lengths.items():
            if length>1:
                self.__dict__[key] = [self.__dict__[key][i] for i in range(self.var_lengths[key]) if i not in index_list]
            kwargs[key] = self.__dict__[key]
        return self.__class__(**kwargs)

    def save(self, foldername):
        for key in self.names:
            torch.save(self.__dict__[key], foldername+key)

    @staticmethod
    def load(foldername):
        from os import walk
        (_, _, filenames) = next(walk(foldername))
        kwargs = {}
        for fn in filenames:
            kwargs[fn] = torch.load(foldername+fn, map_location= device)
        if set(filenames) == set(['parameter', 'field_spatial_coordinates']):
            print('load as QueryDataStruct')
            return QueryDataStruct(**kwargs)
        elif set(filenames) == set(['field']):
            return TargetDataStruct(**kwargs)
        else:
            return DataStruct(**kwargs)

class QueryDataStruct(DataStruct):
    def __init__(self, parameter: List[torch.Tensor],
                    field_spatial_coordinates: List[torch.Tensor]):
        super().__init__(parameter=parameter,
                        field_spatial_coordinates=field_spatial_coordinates)


class TargetDataStruct(DataStruct):
    def __init__(self, field: List[torch.Tensor]):
        super().__init__(field=field)
