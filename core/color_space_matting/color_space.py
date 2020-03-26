#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/3/26 15:09
# @Author  : moiling
# @File    : color_space.py
from sklearn.neighbors.kd_tree import KDTree

from core.data import MattingData
import numpy as np


class ColorSpace:
    """
    Now only use RGB space.
    """
    def __init__(self, data: MattingData):
        self.data = data
        self.unique_color_f, self.unique_color2id_f = self.unique_color(self.data.rgb_f)
        self.unique_color_b, self.unique_color2id_b = self.unique_color(self.data.rgb_b)
        self.all_color = self.all_rgb()
        self.color_space_f, self.color_dist_f = self.rgb_space(self.unique_color_f)
        self.color_space_b, self.color_dist_b = self.rgb_space(self.unique_color_b)

    @staticmethod
    def all_rgb():
        # [256 * 256 * 256, 3]
        return np.reshape(np.mgrid[0:256, 0:256, 0:256], [3, -1]).T

    def rgb_space(self, unique_color):
        """
                         id:   0   1   2   3
        :param unique_color: [C1, C2, C3, C4] shape=(n,3) type=ndarray
        :return:
            rgb_space: nearest unique color id in all rgb space. shape=(256 * 256 * 256,)
            rgb_dist: nearest exist color distance in all rgb space. shape=(256 * 256 * 256,)
        """
        # grid data interp:
        #   rgb_space = griddata(unique_color, np.arange(len(unique_color)), self.all_color, method='nearest')
        # KNN is 10 times faster than gird data interp, and KNN return distance between no color point to nearest color.
        # KNN interp:
        kd_tree = KDTree(unique_color, leaf_size=30, metric='euclidean')
        rgb_dist, rgb_space = kd_tree.query(self.all_color, k=1, return_distance=True)
        return rgb_space, rgb_dist

    @staticmethod
    def unique_color(color):
        """
                                  0   1   2   3   4   5
        :param color: color set [C1, C2, C2, C3, C1, C4]
        :return:
            unique_color = [C1, C2, C3, C4]  type=ndarray
            unique_color2id = [[0, 4], [1, 2], [3], [5]]  type=[ndarray, ndarray, ...] Don't use [c,n], use[c][n]

        eg: get all C2 is in color set:
            unique_color2id[unique_color[1]]  # you must known C2 is unique_color[1].
        """
        unique_color = np.unique(color, axis=0)
        unique_color2id = [(color == i).nonzero()[0] for i in unique_color]
        return unique_color, unique_color2id
