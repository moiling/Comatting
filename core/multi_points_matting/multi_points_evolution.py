#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/4/5 14:52
# @Author  : moiling
# @File    : multi_points_evolution.py
from core.data import MattingData
import numpy as np

from core.fitness import multi_points_fitness, vanilla_fitness


def multi_points_evolution(f_init, b_init, window, data: MattingData, max_fes, center_id):
    if max_fes >= 1e4:
        pop_n = 50
    elif max_fes >= 1e3:
        pop_n = 20
    elif max_fes >= 1e2:
        pop_n = round(max_fes / 50)
    elif max_fes >= 1e1:
        pop_n = round(max_fes / 4)
    else:
        pop_n = 2

    f = np.random.randint(0, data.f_size, [pop_n, len(window)])  # [pop_n, win_number]
    b = np.random.randint(0, data.b_size, [pop_n, len(window)])
    # initial
    f[0, :] = f_init.astype(int)
    b[0, :] = b_init.astype(int)

    # all window
    rgb_u = data.img[window[:, 0], window[:, 1]].astype(int)
    s_u = window
    md_fpu = data.min_dist_f[window[:, 0], window[:, 1]]
    md_bpu = data.min_dist_b[window[:, 0], window[:, 1]]

    v_f = np.zeros([pop_n, len(window)])
    v_b = np.zeros([pop_n, len(window)])

    alpha, fit, c, _, _ = \
        multi_points_fitness(data.rgb_f[f], data.rgb_b[b], data.s_f[f], data.s_b[b], rgb_u, s_u, md_fpu, md_bpu,
                             center_id)

    fes = pop_n * len(window)

    while fes < max_fes:
        shuffle_id = np.random.permutation(pop_n)
        random_pairs = np.array([shuffle_id[:round(pop_n / 2)], shuffle_id[round(pop_n / 2):round(pop_n / 2) * 2]])

        win = fit[random_pairs[0]] < fit[random_pairs[1]]
        winner = random_pairs[0] * win + random_pairs[1] * ~win
        loser = random_pairs[0] * ~win + random_pairs[1] * win

        v_random = np.random.rand(round(pop_n / 2), len(window))
        d_random = np.random.rand(round(pop_n / 2), len(window))
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
            = multi_points_fitness(data.rgb_f[f[loser]], data.rgb_b[b[loser]], data.s_f[f[loser]],
                                   data.s_b[b[loser]], rgb_u, s_u, md_fpu,
                                   md_bpu, center_id)
        fit[loser] = fit_loser
        alpha[loser] = alpha_loser
        c[loser] = c_loser

        fes += round(pop_n / 2) * len(window)

    best_id = np.argmin(fit, axis=0)
    best_f = f[best_id, center_id]
    best_b = b[best_id, center_id]
    best_alpha = alpha[best_id, center_id]
    best_c = c[best_id, center_id]
    best_fit = fit[best_id]

    return best_f, best_b, best_alpha, best_c, best_fit


def multi_points_vanilla_evolution(f_init, b_init, window, data: MattingData, max_fes, center_id):
    if max_fes >= 1e4:
        pop_n = 50
    elif max_fes >= 1e3:
        pop_n = 20
    elif max_fes >= 1e2:
        pop_n = round(max_fes / 50)
    elif max_fes >= 1e1:
        pop_n = round(max_fes / 4)
    else:
        pop_n = 2

    f = np.random.randint(0, data.f_size, [pop_n, len(window)])  # [pop_n, win_number]
    b = np.random.randint(0, data.b_size, [pop_n, len(window)])
    # initial
    f[0, :] = f_init.astype(int)
    b[0, :] = b_init.astype(int)

    # all window
    rgb_u = data.img[window[:, 0], window[:, 1]].astype(int)
    s_u = window
    md_fpu = data.min_dist_f[window[:, 0], window[:, 1]]
    md_bpu = data.min_dist_b[window[:, 0], window[:, 1]]

    v_f = np.zeros([pop_n, len(window)])
    v_b = np.zeros([pop_n, len(window)])

    alpha, fit, c, _, _ = \
        vanilla_fitness(data.rgb_f[f], data.rgb_b[b], data.s_f[f], data.s_b[b], rgb_u, s_u, md_fpu, md_bpu)
    fit = np.sum(fit, axis=1)
    fes = pop_n * len(window)

    while fes < max_fes:
        shuffle_id = np.random.permutation(pop_n)
        random_pairs = np.array([shuffle_id[:round(pop_n / 2)], shuffle_id[round(pop_n / 2):round(pop_n / 2) * 2]])

        win = fit[random_pairs[0]] < fit[random_pairs[1]]
        winner = random_pairs[0] * win + random_pairs[1] * ~win
        loser = random_pairs[0] * ~win + random_pairs[1] * win

        v_random = np.random.rand(round(pop_n / 2), len(window))
        d_random = np.random.rand(round(pop_n / 2), len(window))
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
                              data.s_b[b[loser]], rgb_u, s_u, md_fpu, md_bpu)

        fit_loser = np.sum(fit_loser, axis=1)
        fit[loser] = fit_loser
        alpha[loser] = alpha_loser
        c[loser] = c_loser

        fes += round(pop_n / 2) * len(window)

    best_id = np.argmin(fit, axis=0)
    best_f = f[best_id, center_id]
    best_b = b[best_id, center_id]
    best_alpha = alpha[best_id, center_id]
    best_c = c[best_id, center_id]
    best_fit = fit[best_id]

    return best_f, best_b, best_alpha, best_c, best_fit
