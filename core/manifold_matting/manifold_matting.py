import os
import sys
import time
from enum import Enum

from sklearn.cluster import k_means
from sklearn.cluster import dbscan
from .manifold_evo import manifold_evo, manifold_random
from ..color_space_matting.color_space import ColorSpace
from ..data import MattingData
import numpy as np


class EvolutionType(Enum):
    EVO = 0
    RANDOM = 1


class ManifoldMatting:
    def __init__(self, data: MattingData):
        self.data = data

    def __cluster(self, cluster_size, save_cache=True, save_path='./out/k_means_cluster/'):

        if save_cache and os.path.exists(save_path + self.data.img_name + '_{:.0e}.npz'.format(cluster_size)):
            if self.data.log:
                print('\n{:-^70}\n{:|^70}\n{:-^70}'.format('', ' USE CLUSTER CACHE ', ''))
            cluster_data = np.load(save_path + self.data.img_name + '_{:.0e}.npz'.format(cluster_size))
            center = cluster_data['center']
            cluster_id = cluster_data['cluster_id']
        else:
            if self.data.log:
                print('\n{:-^70}\n{:|^70}\n{:-^70}'.format('', ' COMPUTING CLUSTER ', ''))
            u = np.concatenate([self.data.rgb_u, self.data.s_u], axis=1)
            center, cluster_id, _ = k_means(u, n_clusters=round(self.data.u_size / cluster_size))

        if save_cache and not os.path.exists(save_path + self.data.img_name + '_{:.0e}.npz'.format(cluster_size)):
            if not os.path.exists(save_path):
                os.makedirs(save_path)

            np.savez(save_path + self.data.img_name + '_{:.0e}.npz'.format(cluster_size), center=center, cluster_id=cluster_id)

        for c_id in range(len(center)):
            yield center[c_id], np.where(cluster_id == c_id)[0]

    def matting(self, max_fes, cluster_size, evo_type=EvolutionType.EVO):
        color_space = ColorSpace(self.data, log=self.data.log, use_spatial_space=True)

        f = np.zeros(self.data.u_size, 'int')
        b = np.zeros(self.data.u_size, 'int')
        alpha = np.zeros(self.data.u_size)
        cost_c = np.zeros(self.data.u_size)
        fitness = np.zeros(self.data.u_size)

        time_start = 0
        for cluster_id, cluster in enumerate(self.__cluster(cluster_size)):
            center = cluster[0]
            cluster = cluster[1]
            if self.data.log and cluster_id % 10 == 0:
                time_start = time.time()
            if evo_type == EvolutionType.RANDOM:
                c_f, c_b, c_alpha, c_c, c_fit = manifold_random(center, cluster, self.data, color_space, max_fes * cluster_size)
            else:
                c_f, c_b, c_alpha, c_c, c_fit = manifold_evo(center, cluster, self.data, color_space, max_fes * cluster_size)
            f[cluster] = c_f
            b[cluster] = c_b
            alpha[cluster] = c_alpha
            cost_c[cluster] = c_c
            fitness[cluster] = c_fit

            if self.data.log and cluster_id % 10 == 9:
                print('{:>5.2f}{:>7.2f}s'.format((cluster_id + 1), time.time() - time_start))

        alpha_matte = self.data.trimap.copy()
        alpha_matte[self.data.isu] = alpha * 255
        self.data.alpha_matte = alpha_matte
        self.data.sample_f = f
        self.data.sample_b = b
        self.data.cost_c = cost_c
        self.data.fit = fitness

