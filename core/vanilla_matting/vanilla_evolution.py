#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/3/30 14:51
# @Author  : moiling
# @File    : vanilla_evolution.py
from core.data import MattingData
import numpy as np

from core.fitness import vanilla_fitness


def vanilla_evolution(u, data: MattingData, max_fes):
    if max_fes >= 1e3:
        pop_n = 100
    elif max_fes >= 1e2:
        pop_n = round(max_fes / 10)
    elif max_fes >= 1e1:
        pop_n = round(max_fes / 4)
    else:
        pop_n = 2

    # initial random f/b id, and must be integer.
    f = np.random.randint(0, data.f_size, pop_n)
    b = np.random.randint(0, data.b_size, pop_n)

    rgb_u = data.rgb_u[u]
    s_u = data.s_u[u]
    md_fpu = data.min_dist_f[s_u[0], s_u[1]]
    md_bpu = data.min_dist_b[s_u[0], s_u[1]]

    v_f = np.zeros(pop_n)
    v_b = np.zeros(pop_n)

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

        v_random = np.random.rand(round(pop_n / 2))
        d_random = np.random.rand(round(pop_n / 2))
        v_f[loser] = v_random * v_f[loser] + d_random * (f[winner] - f[loser])
        v_b[loser] = v_random * v_b[loser] + d_random * (b[winner] - b[loser])

        f[loser] = v_f[loser] + f[loser]
        b[loser] = v_b[loser] + b[loser]

        # boundary control
        f[f >= data.f_size] = data.f_size - 1
        b[b >= data.b_size] = data.b_size - 1
        f[f < 0] = 0
        b[b < 0] = 0

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
