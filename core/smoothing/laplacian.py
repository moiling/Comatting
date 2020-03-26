#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/3/24 15:18
# @Author  : moiling
# @File    : laplacian.py
from ..data import MattingData
import numpy as np
import cv2
from scipy.sparse import csr_matrix, spdiags


def laplacian(data: MattingData, epsilon=1e-7, win_size=1):
    image = data.img / 255
    neighbor_size = (win_size * 2 + 1) ** 2
    img_size = data.height * data.width

    # erode known region
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (win_size * 2 + 1, win_size * 2 + 1))
    consts = cv2.erode(np.logical_or(data.isb, data.isf).astype('uint8'), kernel)

    # create indices for all pixels in image, row first.
    img_ind = np.reshape(range(img_size), [data.height, data.width])

    # calculate how many unknown pixels should be estimated.
    est_num = (np.sum(1 - consts[win_size:-win_size, win_size:-win_size]) * (neighbor_size ** 2)).astype(int)
    row_ind = np.zeros(est_num)
    col_ind = np.zeros(est_num)
    val = np.zeros(est_num)

    used_len = 0

    for i in range(win_size, data.height - win_size):
        for j in range(win_size, data.width - win_size):
            if consts[i, j]:
                continue  # ignore any pixels under known regions.

            # get neighbor pixel indices of pixel[i, j].
            win_ind = img_ind[i - win_size:i + win_size + 1, j - win_size: j + win_size + 1]
            win_ind = win_ind.flatten()[np.newaxis]

            # get neighbor pixel matrix from original image.
            win_rgb = image[i - win_size:i + win_size + 1, j - win_size: j + win_size + 1]
            win_rgb = np.reshape(win_rgb, [neighbor_size, data.color_channel])

            # get mean value of each channel
            mean_rgb = np.mean(win_rgb, axis=0)[:, np.newaxis]

            win_var = np.linalg.inv(win_rgb.T.dot(win_rgb) / neighbor_size - mean_rgb.dot(mean_rgb.T) + epsilon / neighbor_size * np.eye(data.color_channel))
            win_rgb = win_rgb - mean_rgb.T

            t_val = (1 + win_rgb.dot(win_var).dot(win_rgb.T)) / neighbor_size

            row_ind[used_len:neighbor_size ** 2 + used_len] = np.reshape(np.tile(win_ind.T, [1, neighbor_size]), [neighbor_size ** 2])
            col_ind[used_len:neighbor_size ** 2 + used_len] = np.reshape(np.tile(win_ind, [neighbor_size, 1]), [neighbor_size ** 2])

            val[used_len:neighbor_size ** 2 + used_len] = t_val.flatten()
            used_len += neighbor_size ** 2

    # cut unused space
    val = val[:used_len]
    row_ind = row_ind[:used_len]
    col_ind = col_ind[:used_len]

    A = csr_matrix((val, (row_ind, col_ind)), shape=(img_size, img_size))
    sum_a = A.sum(axis=1)
    A = spdiags(sum_a.flatten(), 0, img_size, img_size) - A

    return A
