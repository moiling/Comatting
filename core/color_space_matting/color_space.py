#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/3/26 15:09
# @Author  : moiling
# @File    : color_space.py
import os
import sys
import time

from sklearn.neighbors.kd_tree import KDTree
from sklearn import preprocessing
from core.data import MattingData
import numpy as np

from core.spatial_matting.spatial_space import SpatialSpace


class ColorSpace:
    """
    Now only use RGB space.
    """

    def __init__(self, data: MattingData, use_spatial_space=True, log=False, save_cache=True,
                 save_path='./out/color_space/'):
        self.data = data
        self.unique_color_f, self.unique_color2id_f = self.unique_color(self.data.rgb_f)
        self.unique_color_b, self.unique_color2id_b = self.unique_color(self.data.rgb_b)
        self.all_color = self.all_rgb()
        self.use_spatial_space = use_spatial_space
        if self.use_spatial_space:
            self.spatial_space = SpatialSpace(self.data)

        if save_cache and os.path.exists(save_path + data.img_name + '.npz'):
            if log:
                print('\n{:-^70}\n{:|^70}\n{:-^70}'.format('', ' USE COLOR SPACE CACHE ', ''))
            color_space_data = np.load(save_path + data.img_name + '.npz')
            self.color_space_f, self.color_dist_f = color_space_data['color_space_f'], color_space_data['color_dist_f']
            self.color_space_b, self.color_dist_b = color_space_data['color_space_b'], color_space_data['color_dist_b']
        else:
            if log:
                print('\n{:-^70}\n{:|^70}\n{:-^70}'.format('', ' COMPUTING COLOR SPACE ', ''))
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

    def ray_points_rgb_f(self, start_rgb, middle_id, d):
        # d must be [0, 1], list or number.
        middle_rgb = self.data.rgb_u[middle_id]
        distance, k = self.rgb_ray_distance_old(start_rgb, middle_rgb)
        d = np.round(d * distance)[:, np.newaxis]
        rgb_points = start_rgb + d * k

        same_color = np.sum(k != [0, 0, 0], axis=1) == 0
        if np.sum(same_color) > 0:
            if self.use_spatial_space:
                middle_s = self.data.s_u[middle_id]
                if len(middle_s.shape) > 1:
                    rgb_points[same_color] = self.data.rgb_f[self.spatial_space.spatial_space_f[middle_s[same_color, 0], middle_s[same_color, 1]]]
                else:
                    rgb_points[same_color] = self.data.rgb_f[self.spatial_space.spatial_space_f[middle_s[..., 0], middle_s[..., 1]]]
            else:
                rgb_points[same_color] = np.random.randint(0, 255, [3])
        return rgb_points.astype(int)

    def ray_points_uid_f(self, start_rgb, middle_id, d):
        rgb_points = self.ray_points_rgb_f(start_rgb, middle_id, d)
        if rgb_points.ndim == 1:
            rgb_points = rgb_points[np.newaxis, :]
        return self.multi_rgb2unique_color_id(rgb_points, self.color_space_f)

    @staticmethod
    def rgb_ray_distance_old(start_rgb, middle_rgb):
        if start_rgb.ndim == 1:
            start_rgb = start_rgb[np.newaxis, :]

        m_s = middle_rgb - start_rgb
        a = preprocessing.normalize(m_s)
        b = start_rgb

        a_tmp = np.reshape(np.tile(a, 2), [-1, 2, 3])
        b_tmp = np.reshape(np.tile(b, 2), [-1, 2, 3])
        # 6 plane function: [[r, g, b], [r, g, b]] => r=0, r=255, g=0, g=255, ...
        #  [0, 255]     [r_a]       [r_b]
        #  [0, 255]  =  [g_a] * X + [g_b]
        #  [0, 255]     [b_a]       [b_b]
        #
        #       [x_1, x_2]
        #  X =  [x_3, x_4]  => 6 plane intersection distance x，choose the min one that is first intersect.
        #       [x_5, x_6]
        surfaces = np.array([[[0, 0, 0], [255, 255, 255]]])
        x = (surfaces - b_tmp) / (a_tmp + 0.0001)
        x[x <= 0] = 1e5
        x = np.min(np.reshape(x, [len(a), -1]), axis=1)
        # intersection = a * x + b
        return x, a

    @staticmethod
    def rgb_ray_distance(start_rgb, middle_rgb):
        # B => start_rgb, I => middle_rgb
        # where vector BI > 0 => ray BI may intersect the plane(255)
        #              BI < 0 =>                      the plane(0)
        #                 R   G   B
        #     e.g. BI = [10, -20, 0] => may intersect the plane(R=255) or the plane(G=0)
        # and each ray could intersect less than 3 plane(intersect point) => key point => K
        # the MAX axis of (BI ./ BK) is the intersect plane.
        #     e.g. B = [230, 40, 10], I = [240, 20, 10], K = [255, 0, 255],
        #          BI = [10, -20, 0], BK = [25, -40, 245]
        #          BI./BK = [2/5, 1/2, 0]
        #          arg MAX(BI./BK) = 1 => so BI should intersect the plane(G=0)
        # so the distance = (BK / normalize(BI))[arg MAX(BI./BK)].
        #
        # if you want get a point in this ray, ak + B, a in (0, d), a is a integer.

        if start_rgb.ndim == 1:
            start_rgb = start_rgb[np.newaxis, :]

        m_s = middle_rgb - start_rgb
        key_point = (m_s > 0) * 255
        plane_axis = np.argmax(m_s / (key_point - start_rgb), axis=1)
        m_s[m_s == 0] = 0.0001

        range_m_s = np.arange(len(m_s))
        k = preprocessing.normalize(m_s)
        distance = (key_point[range_m_s, plane_axis] - start_rgb[range_m_s, plane_axis]) / (
                    k[range_m_s, plane_axis] + 0.0001)

        return distance, k

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

    def u_color2n_id_b4multi_u(self, unique_color_id, u_id):
        unique_color2id = self.unique_color2id_b
        s = self.data.s_b
        nearest_id = np.zeros(len(unique_color_id))

        for i, uc_id in enumerate(unique_color_id):  # one unique color
            if len(unique_color2id[uc_id]) == 1:
                n = 0
            else:
                n = np.argmin(np.sqrt(np.sum(np.square(s[np.array(unique_color2id[uc_id])] - self.data.s_u[u_id[i]]), axis=1)))
            nearest_id[i] = unique_color2id[uc_id][n]

        return nearest_id.astype(int)

    def u_color2n_id_f4multi_u(self, unique_color_id, u_id):
        unique_color2id = self.unique_color2id_f
        s = self.data.s_f
        nearest_id = np.zeros(len(unique_color_id))

        for i, uc_id in enumerate(unique_color_id):  # one unique color
            if len(unique_color2id[uc_id]) == 1:
                n = 0
            else:
                n = np.argmin(np.sqrt(np.sum(np.square(s[np.array(unique_color2id[uc_id])] - self.data.s_u[u_id[i]]), axis=1)))
            nearest_id[i] = unique_color2id[uc_id][n]

        return nearest_id.astype(int)

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

        for i, uc_id in enumerate(unique_color_id):  # one unique color
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
