#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/3/26 13:25
# @Author  : moiling
# @File    : random_matting.py

import time

from ..data import MattingData
import numpy as np

from ..fitness import vanilla_fitness


class RandomMatting:
    def __init__(self, data: MattingData):
        self.data = data

    def matting(self, max_fes):
        sample_f = np.zeros(self.data.u_size, 'int')
        sample_b = np.zeros(self.data.u_size, 'int')
        alpha = np.zeros(self.data.u_size)
        cost_c = np.zeros(self.data.u_size)

        time_start = 0
        for u in range(self.data.u_size):
            if u % 100 == 0:
                time_start = time.time()

            # random
            # f = np.random.randint(0, self.data.f_size, round(max_fes))
            # b = np.random.randint(0, self.data.b_size, round(max_fes))
            # sample = unique random
            random_fnb_id = (np.random.sample(round(max_fes)) * self.data.f_size * self.data.b_size).astype(int)
            f = random_fnb_id % self.data.f_size
            b = random_fnb_id // self.data.f_size

            alpha_tmp, fit, c, _, _ = \
                vanilla_fitness(self.data.rgb_f[f], self.data.rgb_b[b], self.data.s_f[f],
                                self.data.s_b[b], self.data.rgb_u[u], self.data.s_u[u],
                                self.data.min_dist_f[self.data.isu][u], self.data.min_dist_b[self.data.isu][u])

            best_id = np.argmin(fit)
            sample_f[u] = f[best_id]
            sample_b[u] = b[best_id]
            alpha[u] = alpha_tmp[best_id]
            cost_c[u] = c[best_id]

            if u % 100 == 99:
                print('{:>5.2f}%{:>7.2f}s'.format((u + 1) / self.data.u_size * 100, time.time() - time_start))

        alpha_matte = self.data.trimap.copy()
        alpha_matte[self.data.isu] = alpha * 255
        self.data.alpha_matte = alpha_matte
        self.data.sample_f = sample_f
        self.data.sample_b = sample_b
        self.data.cost_c = cost_c
