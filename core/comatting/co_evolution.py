#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/3/23 16:06
# @Author  : moiling
# @File    : co_evolution.py
from ..data import MattingData
from ..fitness import vanilla_fitness
import numpy as np


def co_evolution(f, b, window, data: MattingData, max_fes):
    # f/b id must be integer.
    f = f.astype(int)
    b = b.astype(int)

    pop = len(f)
    rgb_u = data.img[window[:, 0], window[:, 1]].astype(int)
    s_u = window
    md_fpu = data.min_dist_f[window[:, 0], window[:, 1]]
    md_bpu = data.min_dist_b[window[:, 0], window[:, 1]]

    v_f = np.zeros(pop)
    v_b = np.zeros(pop)

    alpha, fit, c, _, _ = \
        vanilla_fitness(data.rgb_f[f], data.rgb_b[b], data.s_f[f], data.s_b[b], rgb_u, s_u, md_fpu, md_bpu)

    best_f = f
    best_b = b
    best_alpha = alpha
    best_fit = fit
    best_c = c

    fes = pop

    while fes < max_fes:
        shuffle_id = np.random.permutation(pop)
        random_pairs = np.array([shuffle_id[:round(pop / 2)], shuffle_id[round(pop / 2):round(pop / 2) * 2]])

        win = fit[random_pairs[0]] < fit[random_pairs[1]]
        winner = random_pairs[0] * win + random_pairs[1] * ~win
        loser = random_pairs[0] * ~win + random_pairs[1] * win

        v_random = np.random.rand(round(pop / 2))
        d_random = np.random.rand(round(pop / 2))
        v_f[loser] = v_random * v_f[loser] + alpha[winner] * d_random * (f[winner] - f[loser])
        v_b[loser] = v_random * v_b[loser] + (1 - alpha[winner]) * d_random * (b[winner] - b[loser])

        f[loser] = f[loser] + v_f[loser]
        b[loser] = b[loser] + v_b[loser]

        # boundary control
        f[f >= data.f_size] = data.f_size - 1
        f[f < 0] = 0
        b[b >= data.b_size] = data.b_size - 1
        b[b < 0] = 0

        alpha_loser, fit_loser, c_loser, _, _ \
            = vanilla_fitness(data.rgb_f[f[loser]], data.rgb_b[b[loser]], data.s_f[f[loser]],
                              data.s_b[b[loser]], rgb_u[loser], s_u[loser], md_fpu[loser],
                              md_bpu[loser])
        fit[loser] = fit_loser
        alpha[loser] = alpha_loser
        c[loser] = c_loser

        better = fit < best_fit
        best_f[better] = f[better]
        best_b[better] = b[better]
        best_alpha[better] = alpha[better]
        best_fit[better] = fit[better]
        best_c[better] = c[better]

        fes += round(pop / 2)

    # 9 x 9
    for i in range(pop):
        f_i = np.tile(f[i], pop)
        b_i = np.tile(b[i], pop)
        alpha_i, fit_i, c_i, _, _ = \
            vanilla_fitness(data.rgb_f[f_i], data.rgb_b[b_i], data.s_f[f_i], data.s_b[b_i], rgb_u, s_u, md_fpu, md_bpu)

        better = fit_i < best_fit
        best_fit[better] = fit_i[better]
        best_f[better] = f_i[better]
        best_b[better] = b_i[better]
        best_alpha[better] = alpha_i[better]
        best_c[better] = c_i[better]

    return best_f, best_b, best_alpha, best_c, best_fit
