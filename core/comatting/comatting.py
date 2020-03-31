#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/3/23 12:42
# @Author  : moiling
# @File    : comatting.py
import time

from ..data import MattingData
import numpy as np

from .co_evolution import co_evolution


class Comatting:
    def __init__(self, data: MattingData):
        self.data = data

    def __windows(self, n):
        """
        Iterator creator: get 9 neighbor pixels axis.
        :param n: number of unknown pixels.
        :return: current unknown pixels' 9 neighbor pixels axis.
        """
        for u_id in range(n):
            y = self.data.s_u[u_id, 0]
            x = self.data.s_u[u_id, 1]
            yield np.array([
                [y - 1 if y - 1 >= 0 else y, x - 1 if x - 1 > 0 else x],
                [y - 1 if y - 1 >= 0 else y, x],
                [y - 1 if y - 1 >= 0 else y, x + 1 if x + 1 < self.data.width else x],
                [y, x - 1 if x - 1 > 0 else x],
                [y, x],
                [y, x + 1 if x + 1 < self.data.width else x],
                [y + 1 if y + 1 < self.data.height else y, x - 1 if x - 1 > 0 else x],
                [y + 1 if y + 1 < self.data.height else y, x],
                [y + 1 if y + 1 < self.data.height else y, x + 1 if x + 1 < self.data.width else x]
            ])

    def matting(self, max_fes):
        # F and B sample per each pixels, -1 means not initial, F/B areas use current F/B id to initial.
        sample_f = np.ones([self.data.height, self.data.width], 'int') * -1
        sample_b = np.ones([self.data.height, self.data.width], 'int') * -1
        sample_f[self.data.isf] = range(self.data.f_size)
        sample_b[self.data.isb] = range(self.data.b_size)
        # alpha = np.zeros(self.data.u_size, 'int')
        alpha = self.data.trimap.copy()
        alpha = alpha.astype(float) / 255

        cost_c = np.zeros([self.data.height, self.data.width], 'int')
        fitness = np.zeros([self.data.height, self.data.width], 'int')

        time_start = 0
        for u_id, window in enumerate(self.__windows(self.data.u_size)):
            if self.data.log and u_id % 100 == 0:
                time_start = time.time()

            f = sample_f[window[:, 0], window[:, 1]]
            b = sample_b[window[:, 0], window[:, 1]]
            # if not initial sample, random it.
            f[f == -1] = np.random.randint(0, self.data.f_size, np.sum(f == -1))
            b[b == -1] = np.random.randint(0, self.data.b_size, np.sum(b == -1))

            f, b, win_alpha, c, fit = co_evolution(f, b, window, self.data, max_fes)
            sample_f[window[:, 0], window[:, 1]] = f
            sample_b[window[:, 0], window[:, 1]] = b
            # alpha[u_id] = win_alpha[4]
            alpha[window[:, 0], window[:, 1]] = win_alpha
            cost_c[window[:, 0], window[:, 1]] = c
            fitness[window[:, 0], window[:, 1]] = fit

            if self.data.log and u_id % 100 == 99:
                print('{:>5.2f}%{:>7.2f}s'.format((u_id + 1) / self.data.u_size * 100, time.time() - time_start))

        alpha[self.data.isf] = 1
        alpha[self.data.isb] = 0
        self.data.alpha_matte = alpha * 255
        self.data.sample_f = sample_f[self.data.isu]
        self.data.sample_b = sample_b[self.data.isu]
        self.data.cost_c = cost_c[self.data.isu]
        self.data.fit = fitness[self.data.isu]

