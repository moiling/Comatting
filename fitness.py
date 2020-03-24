#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/3/23 15:29
# @Author  : moiling
# @File    : fitness.py
import numpy as np


def fitness(rgb_f, rgb_b, s_f, s_b, rgb_u, s_u, min_dist_fpu, min_dist_bpu):
    """
    :params [num, d]
    """
    alpha = np.sum((rgb_u - rgb_b) * (rgb_f - rgb_b), axis=1) / (np.sum(np.square(rgb_f - rgb_b), axis=1) + 0.01)
    alpha[alpha < 0] = 0
    alpha[alpha > 1] = 1
    # Chromatic distortion
    cost_c = np.sqrt(np.sum(np.square(rgb_u - (alpha[:, np.newaxis] * rgb_f + (1 - alpha[:, np.newaxis]) * rgb_b)), axis=1))
    # Spatial cost
    cost_sf = np.sqrt(np.sum(np.square(s_f - s_u), axis=1)) / (min_dist_fpu + 0.01)
    cost_sb = np.sqrt(np.sum(np.square(s_b - s_u), axis=1)) / (min_dist_bpu + 0.01)
    fit = (cost_c + cost_sf + cost_sb)

    return alpha, fit, cost_c, cost_sf, cost_sb
