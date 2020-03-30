#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/3/30 14:50
# @Author  : moiling
# @File    : vanilla_matting.py
import time

from core.data import MattingData
import numpy as np

from .vanilla_evolution import vanilla_evolution


class VanillaMatting:
    def __init__(self, data: MattingData):
        self.data = data

    def matting(self, max_fes):
        sample_f = np.zeros(self.data.u_size, 'int')
        sample_b = np.zeros(self.data.u_size, 'int')
        alpha = np.zeros(self.data.u_size)
        cost_c = np.zeros(self.data.u_size)

        time_start = 0
        for u_id in range(self.data.u_size):
            if u_id % 100 == 0:
                time_start = time.time()
            f, b, alpha_tmp, c = vanilla_evolution(u_id, self.data, max_fes)

            sample_f[u_id] = f
            sample_b[u_id] = b
            alpha[u_id] = alpha_tmp
            cost_c[u_id] = c

            if u_id % 100 == 99:
                print('{:>5.2f}%{:>7.2f}s'.format((u_id + 1) / self.data.u_size * 100, time.time() - time_start))

        alpha_matte = self.data.trimap.copy()
        alpha_matte[self.data.isu] = alpha * 255
        self.data.alpha_matte = alpha_matte
        self.data.sample_f = sample_f
        self.data.sample_b = sample_b
        self.data.cost_c = cost_c

