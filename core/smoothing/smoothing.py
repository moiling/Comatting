#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/3/24 17:31
# @Author  : moiling
# @File    : smoothing.py
from scipy.sparse import spdiags
from scipy.sparse.linalg import spsolve

from ..data import MattingData
from .laplacian import laplacian
import numpy as np


def smoothing(data: MattingData):
    # parameters
    omega = 1e-1
    lamb = 100

    confidence = data.confidence()
    alpha_hat = data.alpha_matte / 255

    img_size = data.height * data.width
    consts = np.logical_or(data.isb, data.isf).astype('uint8')

    # Generate matting Laplacian matrix
    L = laplacian(data)
    D = spdiags(consts.flatten(), 0, img_size, img_size)
    P = spdiags(confidence.flatten() * ~consts.flatten(), 0, img_size, img_size)
    K = lamb * D + omega * P

    # Solve for alpha
    # x = (L + K) \ (K * alpha)  =>  A\B = inv(A)*B   x=A\B => Ax=B => solve(A,B)
    x = spsolve((L + K), K * alpha_hat.flatten())
    alpha = np.reshape(x, [data.height, data.width])
    alpha[alpha > 1] = 1
    alpha[alpha < 0] = 0
    return alpha * 255
