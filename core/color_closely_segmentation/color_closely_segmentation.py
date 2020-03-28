#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/3/28 12:21
# @Author  : moiling
# @File    : color_closely_segmentation.py
import time

from core.color_space_matting.color_space import ColorSpace
from core.data import MattingData
import numpy as np

c_threshold = 2
s_threshold = 50


class ColorCloselySegmentation:
    def __init__(self, data: MattingData):
        self.data = data

    def segment(self):
        # color space.
        color_space = ColorSpace(self.data)
        t = np.zeros(self.data.u_size)

        time_start = 0
        for u_id in range(self.data.u_size):
            time_start = time.time()

            color_u = self.data.rgb_u[u_id]
            nearest_color_dist_f = color_space.nearest_color_dist(color_u, color_space.color_dist_f)
            nearest_color_dist_b = color_space.nearest_color_dist(color_u, color_space.color_dist_b)
            s_f = self.data.s_f[color_space.u_color2n_id_f(
                np.array([color_space.rgb2unique_color_id(color_u, color_space.color_space_f)]), u_id)][0]
            s_b = self.data.s_b[color_space.u_color2n_id_b(
                np.array([color_space.rgb2unique_color_id(color_u, color_space.color_space_b)]), u_id)][0]
            s_u = self.data.s_u[u_id]
            min_d_f = self.data.min_dist_f[self.data.s_u[u_id, 0], self.data.s_u[u_id, 1]]
            min_d_b = self.data.min_dist_b[self.data.s_u[u_id, 0], self.data.s_u[u_id, 1]]
            d_f = np.sqrt(np.sum(np.square(s_f - s_u)))  # / (min_d_f + 0.01)
            d_b = np.sqrt(np.sum(np.square(s_b - s_u)))  # / (min_d_b + 0.01)

            if nearest_color_dist_f <= c_threshold and nearest_color_dist_b <= c_threshold:
                t[u_id] = 128  # U
            elif nearest_color_dist_f <= c_threshold and d_f <= s_threshold:
                t[u_id] = 255  # F
            elif nearest_color_dist_b <= c_threshold and d_b <= s_threshold:
                t[u_id] = 0  # B
            else:
                t[u_id] = 128  # U

            print('{:>5.2f}%{:>7.2f}s,F:{:^20}B:{:^20}DF:{:^20}DB:{:^20}'.format((u_id + 1) / self.data.u_size * 100, time.time() - time_start, nearest_color_dist_f, nearest_color_dist_b, d_f, d_b))

        seg = self.data.trimap.copy()
        seg[self.data.isu] = t
        return seg
