#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/3/23 16:06
# @Author  : moiling
# @File    : evolution.py
from data import MattingData
from fitness import fitness
import numpy as np


def evolution(f, b, win, data: MattingData):
    max_fes = 1e3
    pop = len(f)
    rgb_u = data.img[win[:, 0], win[:, 1]].astype(int)
    s_u = win
    md_fpu = data.min_dist_f[win[:, 0], win[:, 1]]
    md_bpu = data.min_dist_b[win[:, 0], win[:, 1]]

    v_f = np.zeros(pop)
    v_b = np.zeros(pop)

    fit, alpha = fitness(data.rgb_f[f], data.rgb_b[b], data.s_f[f], data.s_b[b], rgb_u, s_u, md_fpu, md_bpu)

    best_f = f
    best_b = b
    best_alpha = alpha
    best_fit = fit

    fes = pop

    while fes < max_fes:
        shuffle_id = np.arange(pop)
        np.random.shuffle(shuffle_id)
        mask = np.array([shuffle_id[:round(pop / 2)], shuffle_id[round(pop / 2):round(pop / 2) * 2]])
        win = fit[mask[0]] < fit[mask[1]]
        lose = np.logical_not(win)
        winner = np.hstack([mask[0][win], mask[1][lose]])
        loser = np.hstack([mask[0][lose], mask[1][win]])

        v_f[loser] = np.random.rand(round(pop / 2)) * v_f[loser] + alpha[winner] * np.random.rand(round(pop / 2)) * (f[winner] - f[loser])
        v_b[loser] = np.random.rand(round(pop / 2)) * v_b[loser] + (1 - alpha[winner]) * np.random.rand(round(pop / 2)) * (f[winner] - f[loser])

        f[loser] = f[loser] + v_f[loser]
        b[loser] = b[loser] + v_b[loser]

        # boundary control
        f[f >= data.f_size] = data.f_size - 1
        f[f < 0] = 0
        b[b >= data.b_size] = data.b_size - 1
        b[b < 0] = 0

        fit_loser, alpha_loser = fitness(data.rgb_f[f[loser]], data.rgb_b[b[loser]], data.s_f[f[loser]],
                                         data.s_b[b[loser]], rgb_u[loser], s_u[loser], md_fpu[loser], md_bpu[loser])
        fit[loser] = fit_loser
        alpha[loser] = alpha_loser

        better = fit < best_fit
        best_f[better] = f[better]
        best_b[better] = b[better]
        best_alpha[better] = alpha[better]
        best_fit[better] = fit[better]

        fes += round(pop / 2)

    # 9 x 9
    for i in range(pop):
        f_i = np.tile(f[i], pop)
        b_i = np.tile(b[i], pop)
        fit_i, alpha_i = fitness(data.rgb_f[f_i], data.rgb_b[b_i], data.s_f[f_i], data.s_b[b_i],
                                 rgb_u, s_u, md_fpu, md_bpu)
        better = fit_i < best_fit
        best_fit[better] = fit_i[better]
        best_f[better] = f_i[better]
        best_b[better] = b_i[better]
        best_alpha[better] = alpha_i[better]

    return best_f, best_b, best_alpha
