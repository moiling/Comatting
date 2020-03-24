#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/3/23 14:40
# @Author  : moiling
# @File    : data.py
import numpy as np
import cv2
from scipy.ndimage import distance_transform_edt


class MattingData:
    def __init__(self, img_url, trimap_url, img_name):
        self.img_name = img_name
        self.img_url = img_url
        self.trimap_url = trimap_url
        self.img = cv2.imread(img_url)  # [height, width, 3-channels], uint8, 0-255
        self.trimap = cv2.imread(trimap_url, cv2.IMREAD_GRAYSCALE)  # [height, width], uint8, 0-255

        if self.img is None:
            raise Exception('ERROR: No image. Please check your image path {}.'.format(img_url))
        if self.trimap is None:
            raise Exception('ERROR: No trimap. Please check your trimap path {}.'.format(trimap_url))
        if self.img.shape[:2] != self.trimap.shape:
            raise Exception('ERROR: The trimap isn\'t fit the image. Please check: img {}, trimap {}'
                            .format(self.img.shape[:2], self.trimap.shape))

        # is:[height, width], bool; rgb:[size, 3-channels], int64, 0-255; s:[size, 2-axis], int64
        self.isf, self.isb, self.isu, self.rgb_f, self.rgb_b, self.rgb_u, self.s_f, self.s_b, self.s_u = self.__tri_info()
        self.height, self.width, self.color_channel = self.img.shape[0], self.img.shape[1], self.img.shape[2]
        self.u_size, self.f_size, self.b_size = len(self.rgb_u), len(self.rgb_f), len(self.rgb_b)
        # min distance to F\B.
        self.min_dist_f = distance_transform_edt(np.logical_not(self.isf))
        self.min_dist_b = distance_transform_edt(np.logical_not(self.isb))
        # [u_size, 1-id]
        self.sample_f, self.sample_b = [], []
        self.img_f, self.img_b = self.img.copy(), self.img.copy()
        self.cost_c = np.zeros(self.u_size)
        # result
        self.alpha_matte = []
        self.alpha_matte_smoothed = []

    def __tri_info(self):
        # [height, width], bool
        isf = self.trimap == 255
        isb = self.trimap == 0
        isu = np.logical_not(np.logical_or(isf, isb))
        # [size, 3-channels], int64, 0-255
        rgb_f = self.img[isf].astype(int)
        rgb_b = self.img[isb].astype(int)
        rgb_u = self.img[isu].astype(int)
        # [size, 2-axis], int64
        s_f = np.array(np.where(isf)).T
        s_b = np.array(np.where(isb)).T
        s_u = np.array(np.where(isu)).T

        return isf, isb, isu, rgb_f, rgb_b, rgb_u, s_f, s_b, s_u

    def img_fnb(self):
        self.img_f[self.s_u[:, 0], self.s_u[:, 1]] = self.rgb_f[self.sample_f]
        self.img_b[self.s_u[:, 0], self.s_u[:, 1]] = self.rgb_b[self.sample_b]

    def confidence(self):
        mu = 1
        f = np.ones([self.height, self.width])
        f[self.isu] = np.exp(-self.cost_c / 2 * mu ** 2)
        return f
