#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/3/26 15:09
# @Author  : moiling
# @File    : color_space.py
import os
import sys
import time

from sklearn.neighbors.kd_tree import KDTree

from core.data import MattingData
import numpy as np


class ColorSpace:
    """
    Now only use RGB space.
    """

    def __init__(self, data: MattingData, log=False, save_cache=False):
        self.data = data
        self.unique_color_f, self.unique_color2id_f = self.unique_color(self.data.rgb_f)
        self.unique_color_b, self.unique_color2id_b = self.unique_color(self.data.rgb_b)
        self.all_color = self.all_rgb()

        save_path = './out/color_space/'
        if save_cache and os.path.exists(save_path + data.img_name + '.npz'):
            if log:
                print('{:-^70}\n{:|^70}\n{:-^70}'.format('', ' USE COLOR SPACE CACHE ', ''))
            color_space_data = np.load(save_path + data.img_name + '.npz')
            self.color_space_f, self.color_dist_f = color_space_data['color_space_f'], color_space_data['color_dist_f']
            self.color_space_b, self.color_dist_b = color_space_data['color_space_b'], color_space_data['color_dist_b']
        else:
            if log:
                print('{:-^70}\n{:|^70}\n{:-^70}'.format('', ' COMPUTING COLOR SPACE ', ''))
                print('|{:^6}|{:^30}|{:^30}|\n{:-^70}'.format('', 'TIME USED/MIN', 'MEMORY USED/MB', ''))
                print('|{:^6}|'.format('F'), end='')
            t_start = time.time()
            self.color_space_f, self.color_dist_f = self.rgb_space(self.unique_color_f)
            if log:
                print('{:^30.3f}|{:^30.3f}|\n{:-^70}'.format((time.time() - t_start) / 60, (
                        sys.getsizeof(self.color_space_f) + sys.getsizeof(self.color_dist_f)) / 1024 / 1024, ''))
                print('|{:^6}|'.format('B'), end='')
            t_start = time.time()
            self.color_space_b, self.color_dist_b = self.rgb_space(self.unique_color_b)
            if log:
                print('{:^30.3f}|{:^30.3f}|\n{:-^70}'.format((time.time() - t_start) / 60, (
                        sys.getsizeof(self.color_space_b) + sys.getsizeof(self.color_dist_b)) / 1024 / 1024, ''))

        if save_cache and not os.path.exists(save_path + data.img_name + '.npz'):
            if not os.path.exists(save_path):
                os.makedirs(save_path)

            np.savez(save_path + data.img_name + '.npz',
                     color_dist_f=self.color_dist_f, color_space_f=self.color_space_f,
                     color_dist_b=self.color_dist_b, color_space_b=self.color_space_b)

    @staticmethod
    def multi_rgb2unique_color_id(rgb, rgb_space):
        rgb[np.logical_or(np.min(rgb, axis=1) < 0, np.max(rgb, axis=1) > 255)] = [255, 255, 256]
        return rgb_space[rgb[:, 0] * 256 * 256 + rgb[:, 1] * 256 + rgb[:, 2]]

    @staticmethod
    def rgb2unique_color_id(rgb, rgb_space):
        return rgb_space[rgb[0] * 256 * 256 + rgb[1] * 256 + rgb[2]]

    @staticmethod
    def nearest_color_dist(rgb, rgb_dist):
        return rgb_dist[rgb[0] * 256 * 256 + rgb[1] * 256 + rgb[2]]

    @staticmethod
    def multi_nearest_color_dist(rgb, rgb_dist):
        rgb[np.logical_or(np.min(rgb, axis=1) < 0, np.max(rgb, axis=1) > 255)] = [255, 255, 256]
        return rgb_dist[rgb[:, 0] * 256 * 256 + rgb[:, 1] * 256 + rgb[:, 2]]

    def u_color2n_id_f(self, unique_color_id, u):
        return self.uid2nid(unique_color_id, self.unique_color2id_f, self.data.s_f, self.data.s_u[u])

    def u_color2n_id_b(self, unique_color_id, u):
        return self.uid2nid(unique_color_id, self.unique_color2id_b, self.data.s_b, self.data.s_u[u])

    @staticmethod
    def uid2nid(unique_color_id, unique_color2id, s, s_u):
        """
        :param unique_color_id: such as unique_color_f_id, shape=(uc_n,)
        :param unique_color2id: such as unique_color2id_f
        :param s    : such as data.s_f, shape=(f_n, d)
        :param s_u  : only one u, shape=(1, d)
        :return: the nearest color id in rgb_f of the given unique color. shape=(uc_n,)
        """
        nearest_id = np.zeros(len(unique_color_id))

        for i, uc_id in enumerate(unique_color_id):   # one unique color
            if len(unique_color2id[uc_id]) == 1:
                n = 0
            else:
                n = np.argmin(np.sqrt(np.sum(np.square(s[np.array(unique_color2id[uc_id])] - s_u), axis=1)))
            nearest_id[i] = unique_color2id[uc_id][n]

        return nearest_id.astype(int)

    @staticmethod
    def all_rgb():
        # [256 * 256 * 256, 3]
        return np.reshape(np.mgrid[0:256, 0:256, 0:256], [3, -1]).T

    def rgb_space(self, unique_color, overflow_handling=True):
        """
        :param overflow_handling:   if color overflow the rgb space, use [255, 255, 256] to defined overflowed color.
                         id:          0   1   2   3
        :param unique_color:        [C1, C2, C3, C4] shape=(n,3) type=ndarray
        :return:
            rgb_space:  nearest unique color **ID** in all rgb space. shape=(256 * 256 * 256,)
                        if handling overflow color, shape=(256 * 256 * 257,)
            rgb_dist:   nearest exist color distance in all rgb space. shape=(256 * 256 * 256,)
                        if handling overflow color, shape=(256 * 256 * 257,)
        """
        # grid data: rgb_space = griddata(unique_color, np.arange(len(unique_color)), self.all_color, method='nearest')
        # KNN is 10 times faster than gird data interp, and KNN return distance between no color point to nearest color.
        kd_tree = KDTree(unique_color, leaf_size=30, metric='euclidean')
        rgb_dist, rgb_space = kd_tree.query(self.all_color, k=1, return_distance=True)
        if overflow_handling:
            rgb_dist = np.append(rgb_dist, 1e10)  # defined by a large distance.
            rgb_space = np.append(rgb_space, 0)
        return rgb_space, rgb_dist

    @staticmethod
    def unique_color(color, return_inverse=False):
        """
                                  0   1   2   3   4   5
        :param color: color set [C1, C2, C2, C3, C1, C4]
        :param return_inverse:
        :return:
            unique_color = [C1, C2, C3, C4]  type=ndarray
            unique_color2id = [[0, 4], [1, 2], [3], [5]]  type=[[], [], ...] Don't use [c,n], use[c][n]

        eg: get all C2 is in color set:
            unique_color2id[unique_color[1]]  # you must known C2 is unique_color[1].
        """
        unique_color, id2unique_color = np.unique(color, return_inverse=True, axis=0)
        # too slow: => unique_color2id = [(color == i).nonzero()[0] for i in unique_color]
        unique_color2id = [[] for _ in range(len(unique_color))]
        for c_id in range(len(color)):
            unique_color2id[id2unique_color[c_id]].append(c_id)

        ret = (unique_color, unique_color2id,)
        if return_inverse:
            ret += (id2unique_color,)
        return ret
