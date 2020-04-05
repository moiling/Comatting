#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/3/30 16:21
# @Author  : moiling
# @File    : loss.py
from core.data import MattingData
import numpy as np


def sad_loss(data: MattingData, gt):
    return np.sum(np.abs(data.alpha_matte[data.isu].astype(float) - gt[data.isu].astype(float)) / 255) / 1000


def mse_loss(data: MattingData, gt):
    return np.sum(
        np.square((data.alpha_matte[data.isu].astype(float) - gt[data.isu].astype(float)) / 255)) / data.u_size
