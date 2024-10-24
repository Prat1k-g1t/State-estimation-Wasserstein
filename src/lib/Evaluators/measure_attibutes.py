#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 02 12:02:55 2024

@author: prai
"""

#TODO: Need to compute cell centers and add interpolation polynomials

class attributes():
    def __init__(self, measure: torch.Tensor,
                 field_coordinates: torch.Tensor):
        
        self.measure = measure
        self.field_coordinates = field_coordinates

    @staticmethod
    def cdf(self, x):

        res = 0.

        h = abs(self.field_coordinates[0]-self.field_coordinates[1])

        ulimit = x - self.field_coordinates[0]

        for i in range(int(ulimit/h)):
            res += measure[i]

        return res

    @staticmethod
    def icdf(self, s):

        res = 0.

        h = abs(self.field_coordinates[0]-self.field_coordinates[1])

        for i in range(self.field_coordinates.shape[-1]):

            if self.measure[i] <= s:
                pass
            else:
                break

        res = self.field_coordinates[0]+(i+1)*h