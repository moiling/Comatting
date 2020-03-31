#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/3/30 13:53
# @Author  : moiling
# @File    : spatial_evolution.py
from core.color_space_matting.color_space import ColorSpace
from core.data import MattingData
from core.fitness import vanilla_fitness
from core.spatial_matting.spatial_space import SpatialSpace
import numpy as np


def vanilla_evolution(u, data: MattingData, spatial_space: SpatialSpace, max_fes):
    if max_fes >= 1e3:
        pop_n = 100
    elif max_fes >= 1e2:
        pop_n = round(max_fes / 10)
    elif max_fes >= 1e1:
        pop_n = round(max_fes / 3)
    else:
        pop_n = 2

    # initial random f/b id, and must be integer.
    f = np.random.randint(0, data.f_size, pop_n)
    b = np.random.randint(0, data.b_size, pop_n)

    rgb_u = data.rgb_u[u]
    s_u = data.s_u[u]
    md_fpu = data.min_dist_f[s_u[0], s_u[1]]
    md_bpu = data.min_dist_b[s_u[0], s_u[1]]

    v_f = np.zeros([pop_n, 2])  # [pop_n, [x,y]]
    v_b = np.zeros([pop_n, 2])

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

        # learning in xy space.
        f_xy = data.s_f[f]
        b_xy = data.s_b[b]
        v_f[loser] = v_random * v_f[loser] + alpha[winner, np.newaxis] * d_random * (f_xy[winner] - f_xy[loser])
        v_b[loser] = v_random * v_b[loser] + (1 - alpha[winner, np.newaxis]) * d_random * (b_xy[winner] - b_xy[loser])

        f_xy[loser] = v_f[loser] + f_xy[loser]
        b_xy[loser] = v_b[loser] + b_xy[loser]

        # boundary control
        f_xy[f_xy[:, 0] >= data.height, 0] = data.height - 1
        f_xy[f_xy[:, 1] >= data.width, 1] = data.width - 1
        b_xy[b_xy[:, 0] >= data.height, 0] = data.height - 1
        b_xy[b_xy[:, 1] >= data.width, 1] = data.width - 1
        f_xy[f_xy < 0] = 0
        b_xy[b_xy < 0] = 0

        f[loser] = spatial_space.spatial_space_f[f_xy[loser, 0], f_xy[loser, 1]]
        b[loser] = spatial_space.spatial_space_b[b_xy[loser, 0], b_xy[loser, 1]]

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


def unique_color_evolution(u, data: MattingData, spatial_space: SpatialSpace, max_fes):
    pop_n = 100

    # initial random f/b id, and must be integer.
    f_u = np.random.randint(0, spatial_space.uc_f_size, pop_n)  # unique color id
    b_u = np.random.randint(0, spatial_space.uc_b_size, pop_n)
    # color id
    f = ColorSpace.uid2nid(f_u, spatial_space.uc2id_f, data.s_f, data.s_u[u])
    b = ColorSpace.uid2nid(b_u, spatial_space.uc2id_b, data.s_b, data.s_u[u])

    rgb_u = data.rgb_u[u]
    s_u = data.s_u[u]
    md_fpu = data.min_dist_f[s_u[0], s_u[1]]
    md_bpu = data.min_dist_b[s_u[0], s_u[1]]

    v_f = np.zeros([pop_n, 2])  # [pop_n, [x,y]]
    v_b = np.zeros([pop_n, 2])

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

        # learning in xy space.
        f_xy = data.s_f[f]
        b_xy = data.s_b[b]
        v_f[loser] = v_random * v_f[loser] + alpha[winner, np.newaxis] * d_random * (f_xy[winner] - f_xy[loser])
        v_b[loser] = v_random * v_b[loser] + (1 - alpha[winner, np.newaxis]) * d_random * (b_xy[winner] - b_xy[loser])

        f_xy[loser] = v_f[loser] + f_xy[loser]
        b_xy[loser] = v_b[loser] + b_xy[loser]

        # boundary control
        f_xy[f_xy[:, 0] >= data.height, 0] = data.height - 1
        f_xy[f_xy[:, 1] >= data.width, 1] = data.width - 1
        b_xy[b_xy[:, 0] >= data.height, 0] = data.height - 1
        b_xy[b_xy[:, 1] >= data.width, 1] = data.width - 1
        f_xy[f_xy < 0] = 0
        b_xy[b_xy < 0] = 0

        f_u[loser] = spatial_space.id2uc_f[spatial_space.spatial_space_f[f_xy[loser, 0], f_xy[loser, 1]]]
        b_u[loser] = spatial_space.id2uc_b[spatial_space.spatial_space_b[b_xy[loser, 0], b_xy[loser, 1]]]

        f[loser] = ColorSpace.uid2nid(f_u[loser], spatial_space.uc2id_f, data.s_f, data.s_u[u])
        b[loser] = ColorSpace.uid2nid(b_u[loser], spatial_space.uc2id_b, data.s_b, data.s_u[u])

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
