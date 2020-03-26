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
        self.max_compute_epoch = 1e7

    def matting(self, max_fes):
        sample_f = np.zeros(self.data.u_size, 'int')
        sample_b = np.zeros(self.data.u_size, 'int')
        alpha = np.zeros(self.data.u_size)
        cost_c = np.zeros(self.data.u_size)

        cut_u = np.array_split(range(self.data.u_size), max_fes * self.data.u_size / self.max_compute_epoch + 1)

        print('epoch:{}'.format(len(cut_u)))
        for us in cut_u:
            time_start = time.time()
            # random
            # f = np.random.randint(0, self.data.f_size, round(max_fes))
            # b = np.random.randint(0, self.data.b_size, round(max_fes))
            # sample = unique random

            # u = [u_n,], rgb_u = [u_n, 3], s_u = [u_n,2], f = [sample_n,u_n], rgb_f = [sample_n,u_n,3]
            random_fnb_id = (np.random.sample([round(max_fes), len(us)]) * self.data.f_size * self.data.b_size).astype(int)
            # each pixels samples can be same, such as below, but it's slow!!
            # random_fnb_id = (np.array(
            #     [np.random.sample(round(max_fes)) for _ in us]).T * self.data.f_size * self.data.b_size).astype(int)
            f = random_fnb_id % self.data.f_size
            b = random_fnb_id // self.data.f_size

            alpha_tmp, fit, c, _, _ = \
                vanilla_fitness(self.data.rgb_f[f], self.data.rgb_b[b], self.data.s_f[f],
                                self.data.s_b[b], self.data.rgb_u[us], self.data.s_u[us],
                                self.data.min_dist_f[self.data.isu][us], self.data.min_dist_b[self.data.isu][us])

            best_id = np.argmin(fit, axis=0)
            sample_f[us] = f[best_id, range(len(us))]
            sample_b[us] = b[best_id, range(len(us))]
            alpha[us] = alpha_tmp[best_id, range(len(us))]
            cost_c[us] = c[best_id, range(len(us))]

            print('{:>7.2f}s'.format(time.time() - time_start))

        alpha_matte = self.data.trimap.copy()
        alpha_matte[self.data.isu] = alpha * 255
        self.data.alpha_matte = alpha_matte
        self.data.sample_f = sample_f
        self.data.sample_b = sample_b
        self.data.cost_c = cost_c
