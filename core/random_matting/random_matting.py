#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/3/26 13:25
# @Author  : moiling
# @File    : random_matting.py

import time

from ..color_space_matting.color_space import ColorSpace
from ..data import MattingData
import numpy as np

from ..fitness import vanilla_fitness


class RandomMatting:
    def __init__(self, data: MattingData):
        self.data = data
        self.max_epoch_points = 1e7
        self.sample_f = np.zeros(self.data.u_size, 'int')
        self.sample_b = np.zeros(self.data.u_size, 'int')
        self.alpha = np.zeros(self.data.u_size)
        self.cost_c = np.zeros(self.data.u_size)
        self.fitness = np.zeros(self.data.u_size)

    def matting(self, max_fes, unique_color=False):
        if not unique_color:
            self.all_color_matting(max_fes)
        else:
            self.unique_color_matting(max_fes)

        alpha_matte = self.data.trimap.copy()
        alpha_matte[self.data.isu] = self.alpha * 255
        self.data.alpha_matte = alpha_matte
        self.data.sample_f = self.sample_f
        self.data.sample_b = self.sample_b
        self.data.cost_c = self.cost_c
        self.data.fit = self.fitness

    def all_color_matting(self, max_fes):
        cut_u = np.array_split(range(self.data.u_size), max_fes * self.data.u_size / self.max_epoch_points + 1)

        if self.data.log:
            print('epoch:{}'.format(len(cut_u)))
        for us in cut_u:
            time_start = time.time()
            # random
            # f = np.random.randint(0, self.data.f_size, round(max_fes))
            # b = np.random.randint(0, self.data.b_size, round(max_fes))
            # sample = unique random

            # u = [u_n,], rgb_u = [u_n, 3], s_u = [u_n,2], f = [sample_n,u_n], rgb_f = [sample_n,u_n,3]
            random_fnb_id = (np.random.sample([round(max_fes), len(us)]) * self.data.f_size * self.data.b_size).astype(
                int)
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
            self.sample_f[us] = f[best_id, range(len(us))]
            self.sample_b[us] = b[best_id, range(len(us))]
            self.alpha[us] = alpha_tmp[best_id, range(len(us))]
            self.cost_c[us] = c[best_id, range(len(us))]
            self.fitness[us] = fit[best_id, range(len(us))]

            if self.data.log:
                print('{:>7.2f}s'.format(time.time() - time_start))

    def unique_color_matting(self, max_fes):
        uc_f, uc2id_f = ColorSpace.unique_color(self.data.rgb_f)
        uc_b, uc2id_b = ColorSpace.unique_color(self.data.rgb_b)
        uc_f_size = len(uc_f)
        uc_b_size = len(uc_b)

        for u in range(self.data.u_size):
            time_start = time.time()

            random_fnb_id = (np.random.sample(round(max_fes)) * uc_f_size * uc_b_size).astype(int)
            f_u = random_fnb_id % uc_f_size
            b_u = random_fnb_id // uc_f_size
            f = ColorSpace.uid2nid(f_u, uc2id_f, self.data.s_f, self.data.s_u[u])
            b = ColorSpace.uid2nid(b_u, uc2id_b, self.data.s_b, self.data.s_u[u])

            alpha_tmp, fit, c, _, _ = \
                vanilla_fitness(self.data.rgb_f[f], self.data.rgb_b[b], self.data.s_f[f],
                                self.data.s_b[b], self.data.rgb_u[u], self.data.s_u[u],
                                self.data.min_dist_f[self.data.isu][u], self.data.min_dist_b[self.data.isu][u])

            best_id = np.argmin(fit)
            self.sample_f[u] = f[best_id]
            self.sample_b[u] = b[best_id]
            self.alpha[u] = alpha_tmp[best_id]
            self.cost_c[u] = c[best_id]
            self.fitness[u] = fit[best_id]

            if self.data.log:
                print('{:>7.2f}s'.format(time.time() - time_start))