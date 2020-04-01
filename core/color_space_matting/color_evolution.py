#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/3/27 16:20
# @Author  : moiling
# @File    : color_evolution.py
import numpy as np

from .color_space import ColorSpace
from ..data import MattingData
from ..fitness import vanilla_fitness


def vanilla_evolution(u, data: MattingData, color_space: ColorSpace, max_fes):
    """
        f/b:[pop_n, d], u:int
        f_u/b_u id is unique color id.
    """
    if max_fes >= 1e3:
        pop_n = 100
    elif max_fes >= 1e2:
        pop_n = round(max_fes / 10)
    elif max_fes >= 1e1:
        pop_n = round(max_fes / 4)
    else:
        pop_n = 2

    # initial random f/b id, and must be integer.
    f_u = np.random.randint(0, len(color_space.unique_color_f), pop_n)  # unique color id
    b_u = np.random.randint(0, len(color_space.unique_color_b), pop_n)
    f = color_space.u_color2n_id_f(f_u, u)  # color id
    b = color_space.u_color2n_id_b(b_u, u)

    rgb_u = data.rgb_u[u]
    s_u = data.s_u[u]
    md_fpu = data.min_dist_f[s_u[0], s_u[1]]
    md_bpu = data.min_dist_b[s_u[0], s_u[1]]

    v_f = np.zeros([pop_n, data.color_channel])
    v_b = np.zeros([pop_n, data.color_channel])

    alpha, fit, c, _, _ = \
        vanilla_fitness(data.rgb_f[f], data.rgb_b[b], data.s_f[f], data.s_b[b], rgb_u, s_u, md_fpu, md_bpu)

    fes = pop_n

    # evolution
    while fes < max_fes:
        shuffle_id = np.random.permutation(pop_n)
        random_pairs = np.array([shuffle_id[:round(pop_n / 2)], shuffle_id[round(pop_n / 2):round(pop_n / 2) * 2]])

        win = fit[random_pairs[0]] < fit[random_pairs[1]]
        winner = random_pairs[0] * win + random_pairs[1] * ~win
        loser = random_pairs[0] * ~win + random_pairs[1] * win

        v_random = np.random.rand(round(pop_n / 2), 1)
        d_random = np.random.rand(round(pop_n / 2), 1)

        # learning in rgb space.
        f_u_rgb = color_space.unique_color_f[f_u].astype(int)
        b_u_rgb = color_space.unique_color_b[b_u].astype(int)

        # if initial alpha = [0,0,0,0,...,0] => don't move
        v_f[loser] = v_random * v_f[loser] + alpha[winner, np.newaxis] * d_random * (f_u_rgb[winner] - f_u_rgb[loser])
        v_b[loser] = v_random * v_b[loser] + (1 - alpha[winner, np.newaxis]) * d_random * (b_u_rgb[winner] - b_u_rgb[loser])

        f_u_rgb[loser] = v_f[loser] + f_u_rgb[loser]
        b_u_rgb[loser] = v_b[loser] + b_u_rgb[loser]

        # boundary control
        f_u_rgb[f_u_rgb > 255] = 255
        f_u_rgb[f_u_rgb < 0] = 0
        b_u_rgb[b_u_rgb > 255] = 255
        b_u_rgb[b_u_rgb < 0] = 0

        f_u[loser] = color_space.multi_rgb2unique_color_id(f_u_rgb[loser], color_space.color_space_f)
        b_u[loser] = color_space.multi_rgb2unique_color_id(b_u_rgb[loser], color_space.color_space_b)

        f[loser] = color_space.u_color2n_id_f(f_u[loser], u)
        b[loser] = color_space.u_color2n_id_b(b_u[loser], u)

        alpha_loser, fit_loser, c_loser, _, _ \
            = vanilla_fitness(data.rgb_f[f[loser]], data.rgb_b[b[loser]], data.s_f[f[loser]],
                              data.s_b[b[loser]], rgb_u, s_u, md_fpu, md_bpu)
        fit[loser] = fit_loser
        alpha[loser] = alpha_loser
        c[loser] = c_loser

        fes += round(pop_n / 2)

    best_id = np.argmin(fit, axis=0)
    best_f = f[best_id]
    best_b = b[best_id]
    best_alpha = alpha[best_id]
    best_c = c[best_id]
    best_fit = fit[best_id]

    return best_f, best_b, best_alpha, best_c, best_fit
