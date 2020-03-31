#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/3/26 15:05
# @Author  : moiling
# @File    : color_space_matting.py

import time

from .color_space import ColorSpace
from ..data import MattingData
from .color_evolution import vanilla_evolution
import numpy as np


class ColorSpaceMatting:
    def __init__(self, data: MattingData):
        self.data = data

    def matting(self, max_fes):
        # color space.
        color_space = ColorSpace(self.data, log=self.data.log)

        sample_f = np.zeros(self.data.u_size, 'int')
        sample_b = np.zeros(self.data.u_size, 'int')
        alpha = np.zeros(self.data.u_size)
        cost_c = np.zeros(self.data.u_size)
        fitness = np.zeros(self.data.u_size)

        time_start = 0
        for u_id in range(self.data.u_size):
            if self.data.log and u_id % 100 == 0:
                time_start = time.time()
            sample_f[u_id], sample_b[u_id], alpha[u_id], cost_c[u_id], fitness[u_id] = \
                vanilla_evolution(u_id, self.data, color_space, max_fes)

            if self.data.log and u_id % 100 == 99:
                print('{:>5.2f}%{:>7.2f}s'.format((u_id + 1) / self.data.u_size * 100, time.time() - time_start))

        alpha_matte = self.data.trimap.copy()
        alpha_matte[self.data.isu] = alpha * 255
        self.data.alpha_matte = alpha_matte
        self.data.sample_f = sample_f
        self.data.sample_b = sample_b
        self.data.cost_c = cost_c
        self.data.fit = fitness
