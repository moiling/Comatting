#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/3/23 15:29
# @Author  : moiling
# @File    : fitness.py
import numpy as np


def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])


def multi_points_fitness(rgb_f, rgb_b, s_f, s_b, rgb_u, s_u, min_dist_fpu, min_dist_bpu, center_id):
    # center_id: [3x3] => 4
    value_axis = rgb_f.ndim - 1
    if value_axis == 1:
        rgb_f = rgb_f[np.newaxis, :]
        rgb_b = rgb_b[np.newaxis, :]
        s_f = s_f[np.newaxis, :]
        s_b = s_b[np.newaxis, :]

    alpha, fit, cost_c, cost_sf, cost_sb = vanilla_fitness(rgb_f, rgb_b, s_f, s_b, rgb_u, s_u, min_dist_fpu, min_dist_bpu)
    #
    #   fit_i = sf_i + sb_i + c_i
    #         + min(
    #               \sum_{n \in N_i}||rgb_f_i - rgb_f_n||_2,
    #               \sum_{n \in N_i}||rgb_b_i - rgb_b_n||_2,
    #               \sum_{n \in N_i}||\alpha_i - \alpha_n||_2
    #              )
    #   *：rgb -> gray [0, 1]
    #
    gray_f = rgb2gray(rgb_f) / 255
    gray_b = rgb2gray(rgb_b) / 255
    f_error = np.sum(np.sqrt(np.sum(np.square(gray_f[:, center_id][:, None] - gray_f), axis=value_axis - 1)))
    b_error = np.sum(np.sqrt(np.sum(np.square(gray_b[:, center_id][:, None] - gray_b), axis=value_axis - 1)))
    alpha_error = np.sum(np.sqrt(np.sum(np.square(alpha[:, center_id][:, None] - alpha), axis=value_axis - 1)))
    min_error = min(min(f_error, b_error), alpha_error)
    fitness = fit[:, center_id] + min_error
    return alpha, fitness, cost_c, cost_sf, cost_sb


def vanilla_fitness(rgb_f, rgb_b, s_f, s_b, rgb_u, s_u, min_dist_fpu, min_dist_bpu):
    """
    fitness function of global matting.
    :params [num, d] => multi: f/b:[pop_n, u_n, d], u:[u_n, d]
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
            np.sum(np.square(rgb_u - (np.expand_dims(alpha, axis=value_axis) * rgb_f + (
                    1 - np.expand_dims(alpha, axis=value_axis)) * rgb_b)), axis=value_axis))
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
