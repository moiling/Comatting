#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/4/5 14:49
# @Author  : moiling
# @File    : multi_points_matting.py
import time
from enum import Enum

from core.color_space_matting.color_space import ColorSpace
from core.data import MattingData
import numpy as np

from core.fitness import multi_points_f_b_fitness, multi_points_fitness, multi_points_single_fitness
from core.multi_points_matting.multi_points_evolution import multi_points_evolution, multi_points_vanilla_evolution


class EvolutionType(Enum):
    MIN_FB_RGB_ALPHA = 0
    MIN_FB_RGB = 1
    VANILLA = 2
    RANDOM_MIN_FB_RGB_ALPHA = 3
    RANDOM_FB_RGB = 4
    RANDOM_SINGLE = 5


class MultiPointsMatting:
    def __init__(self, data: MattingData):
        self.data = data

    def __windows(self, n):
        X, Y = np.meshgrid(np.arange(self.data.width), np.arange(self.data.height))
        for u_id in range(n):
            y = self.data.s_u[u_id, 0]
            x = self.data.s_u[u_id, 1]
            y_min = max(0, y - 1)
            y_max = min(self.data.height - 1, y + 1)
            x_min = max(0, x - 1)
            x_max = min(self.data.width - 1, x + 1)
            yield np.reshape(np.array([Y[y_min:y_max + 1, x_min:x_max + 1], X[y_min:y_max + 1, x_min:x_max + 1]]),
                             [2, -1]).T

    def matting(self, max_fes, evo_type=EvolutionType.MIN_FB_RGB_ALPHA):
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

        # random
        if evo_type == EvolutionType.RANDOM_MIN_FB_RGB_ALPHA:
            self.random_uc(max_fes, multi_points_fitness)
            return
        if evo_type == EvolutionType.RANDOM_FB_RGB:
            self.random_uc(max_fes, multi_points_f_b_fitness)
            return
        if evo_type == EvolutionType.RANDOM_SINGLE:
            self.random_uc(max_fes, multi_points_single_fitness)
            return

        time_start = 0
        for u_id, window in enumerate(self.__windows(self.data.u_size)):
            if self.data.log and u_id % 100 == 0:
                time_start = time.time()

            f = sample_f[window[:, 0], window[:, 1]]
            b = sample_b[window[:, 0], window[:, 1]]
            # if not initial sample, random it.
            f[f == -1] = np.random.randint(0, self.data.f_size, np.sum(f == -1))
            b[b == -1] = np.random.randint(0, self.data.b_size, np.sum(b == -1))

            if len(window) == 9:
                center_id = 4
            else:
                center_id = np.where(np.logical_and(window[:, 0] == self.data.s_u[u_id, 0],
                                                    window[:, 1] == self.data.s_u[u_id, 1]))[0][0]
            if evo_type == EvolutionType.VANILLA:
                f, b, center_alpha, c, fit = multi_points_vanilla_evolution(f, b, window, self.data, max_fes, center_id)
            elif evo_type == EvolutionType.MIN_FB_RGB:
                f, b, center_alpha, c, fit = multi_points_evolution(f, b, window, self.data, max_fes, center_id,
                                                                    fitness_func=multi_points_f_b_fitness)
            else:
                f, b, center_alpha, c, fit = multi_points_evolution(f, b, window, self.data, max_fes, center_id)

            sample_f[window[center_id, 0], window[center_id, 1]] = f
            sample_b[window[center_id, 0], window[center_id, 1]] = b
            # alpha[u_id] = win_alpha[4]
            alpha[window[center_id, 0], window[center_id, 1]] = center_alpha
            cost_c[window[center_id, 0], window[center_id, 1]] = c
            fitness[window[center_id, 0], window[center_id, 1]] = fit

            if self.data.log and u_id % 100 == 99:
                print('{:>5.2f}%{:>7.2f}s'.format((u_id + 1) / self.data.u_size * 100, time.time() - time_start))

        alpha[self.data.isf] = 1
        alpha[self.data.isb] = 0
        self.data.alpha_matte = alpha * 255
        self.data.alpha_matte = self.data.alpha_matte.astype(int)
        self.data.sample_f = sample_f[self.data.isu]
        self.data.sample_b = sample_b[self.data.isu]
        self.data.cost_c = cost_c[self.data.isu]
        self.data.fit = fitness[self.data.isu]

    def random_uc(self, max_fes, fitness_function):
        sample_f = np.ones([self.data.height, self.data.width], 'int') * -1
        sample_b = np.ones([self.data.height, self.data.width], 'int') * -1
        sample_f[self.data.isf] = range(self.data.f_size)
        sample_b[self.data.isb] = range(self.data.b_size)
        alpha = self.data.trimap.copy()
        alpha = alpha.astype(float) / 255

        cost_c = np.zeros([self.data.height, self.data.width], 'int')
        fitness = np.zeros([self.data.height, self.data.width], 'int')

        for u_id, window in enumerate(self.__windows(self.data.u_size)):
            time_start = time.time()
            if len(window) == 9:
                center_id = 4
            else:
                center_id = np.where(np.logical_and(window[:, 0] == self.data.s_u[u_id, 0],
                                                    window[:, 1] == self.data.s_u[u_id, 1]))[0][0]
            random_fnb_id = (np.random.sample(
                [round(max_fes / len(window)), len(window)]) * self.data.f_size * self.data.b_size).astype(int)

            rgb_u = self.data.img[window[:, 0], window[:, 1]].astype(int)
            s_u = window
            md_fpu = self.data.min_dist_f[window[:, 0], window[:, 1]]
            md_bpu = self.data.min_dist_b[window[:, 0], window[:, 1]]

            f = random_fnb_id % self.data.f_size
            b = random_fnb_id // self.data.f_size

            alpha_tmp, fit, c, _, _ = \
                fitness_function(self.data.rgb_f[f], self.data.rgb_b[b], self.data.s_f[f],
                                 self.data.s_b[b], rgb_u, s_u, md_fpu, md_bpu, center_id)

            best_id = np.argmin(fit, axis=0)
            sample_f[window[center_id, 0], window[center_id, 1]] = f[best_id, center_id]
            sample_b[window[center_id, 0], window[center_id, 1]] = b[best_id, center_id]
            alpha[window[center_id, 0], window[center_id, 1]] = alpha_tmp[best_id, center_id]
            cost_c[window[center_id, 0], window[center_id, 1]] = c[best_id, center_id]
            fitness[window[center_id, 0], window[center_id, 1]] = fit[best_id]

            if self.data.log:
                print('{:>7.2f}s'.format(time.time() - time_start))

        alpha[self.data.isf] = 1
        alpha[self.data.isb] = 0
        self.data.alpha_matte = alpha * 255
        self.data.alpha_matte = self.data.alpha_matte.astype(int)
        self.data.sample_f = sample_f[self.data.isu]
        self.data.sample_b = sample_b[self.data.isu]
        self.data.cost_c = cost_c[self.data.isu]
        self.data.fit = fitness[self.data.isu]
