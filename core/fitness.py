#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/3/23 15:29
# @Author  : moiling
# @File    : fitness.py
import numpy as np


def vanilla_fitness(rgb_f, rgb_b, s_f, s_b, rgb_u, s_u, min_dist_fpu, min_dist_bpu):
    """
    fitness function of global matting.
    :params [num, d]
    """
    # get rgb/s' axis. rgb and s' axis must be same.
    if rgb_f.ndim != rgb_b.ndim or rgb_f.ndim != s_f.ndim or rgb_f.ndim != s_b.ndim:
        raise Exception('ERROR: rgb/s\' axis. rgb and s\' axis must be same')
    value_axis = rgb_f.ndim - 1

    if value_axis > 0:
        alpha = np.sum((rgb_u - rgb_b) * (rgb_f - rgb_b), axis=value_axis) / (
                np.sum(np.square(rgb_f - rgb_b), axis=value_axis) + 0.01)

        alpha[alpha < 0] = 0
        alpha[alpha > 1] = 1
        # Chromatic distortion
        cost_c = np.sqrt(
            np.sum(np.square(rgb_u - (alpha[:, np.newaxis] * rgb_f + (1 - alpha[:, np.newaxis]) * rgb_b)),
                   axis=value_axis))
        # Spatial cost
        cost_sf = np.sqrt(np.sum(np.square(s_f - s_u), axis=value_axis)) / (min_dist_fpu + 0.01)
        cost_sb = np.sqrt(np.sum(np.square(s_b - s_u), axis=value_axis)) / (min_dist_bpu + 0.01)
        fit = (cost_c + cost_sf + cost_sb)

        return alpha, fit, cost_c, cost_sf, cost_sb

    else:  # only 1 point
        alpha = np.sum((rgb_u - rgb_b) * (rgb_f - rgb_b)) / (np.sum(np.square(rgb_f - rgb_b)) + 0.01)
        alpha = max(min(alpha, 1.0), 0.0)
        cost_c = np.sqrt(np.sum(np.square(rgb_u - (alpha * rgb_f + (1 - alpha) * rgb_b))))
        cost_sf = np.sqrt(np.sum(np.square(s_f - s_u))) / (min_dist_fpu + 0.01)
        cost_sb = np.sqrt(np.sum(np.square(s_b - s_u))) / (min_dist_bpu + 0.01)
        fit = (cost_c + cost_sf + cost_sb)

        return alpha, fit, cost_c, cost_sf, cost_sb
