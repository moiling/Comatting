#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/3/29 15:18
# @Author  : moiling
# @File    : spatial_space.py
from sklearn.neighbors.kd_tree import KDTree

from core.color_space_matting.color_space import ColorSpace
from core.data import MattingData
import numpy as np


class SpatialSpace:
    def __init__(self, data: MattingData, unique_color=False):
        self.data = data
        self.spatial_space_f, self.spatial_space_b = self.xy_space()
        self.id2uc_f, self.uc2id_f = [], []
        self.id2uc_b, self.uc2id_b = [], []
        self.uc_f_size, self.uc_b_size = 0, 0
        if unique_color:
            self.set_unique_color()

    def xy_space(self):
        xy_space_f = np.zeros([self.data.height, self.data.width])
        xy_space_b = np.zeros([self.data.height, self.data.width])
        xy_space_f[self.data.isf] = np.arange(self.data.f_size)
        xy_space_f[self.data.isb] = self.nearest_xy_id(self.data.s_f, self.data.s_b)
        xy_space_f[self.data.isu] = self.nearest_xy_id(self.data.s_f, self.data.s_u)
        xy_space_b[self.data.isb] = np.arange(self.data.b_size)
        xy_space_b[self.data.isf] = self.nearest_xy_id(self.data.s_b, self.data.s_f)
        xy_space_b[self.data.isu] = self.nearest_xy_id(self.data.s_b, self.data.s_u)
        return xy_space_f.astype(int), xy_space_b.astype(int)

    @staticmethod
    def nearest_xy_id(known_spatial, unknown_spatial):
        kd_tree = KDTree(known_spatial, leaf_size=30, metric='euclidean')
        xy_dist, xy_space = kd_tree.query(unknown_spatial, k=1, return_distance=True)
        return np.squeeze(xy_space)

    def set_unique_color(self):
        uc_f, self.uc2id_f, self.id2uc_f = ColorSpace.unique_color(self.data.rgb_f, return_inverse=True)
        uc_b, self.uc2id_b, self.id2uc_b = ColorSpace.unique_color(self.data.rgb_b, return_inverse=True)
        self.uc_f_size = len(uc_f)
        self.uc_b_size = len(uc_b)
