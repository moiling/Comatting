#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/3/23 13:12
# @Author  : moiling
# @File    : matting.py
import smoothing
from comatting import Comatting
from data import MattingData


class Matting:
    def __init__(self, img_url, trimap_url, img_name=''):
        self.data = MattingData(img_url, trimap_url, img_name)

    def matting(self, func_name='comatting', max_fes=1e3):
        if func_name == 'comatting':
            Comatting(self.data).matting(max_fes)
        else:
            raise Exception('ERROR: no matting function named {}.'.format(func_name))

    def comatting(self, max_fes=1e3):
        Comatting(self.data).matting(max_fes)
        return self.data.alpha_matte

    def img_fnb(self):
        self.data.img_fnb()
        return self.data.img_f, self.data.img_b

    def smoothing(self):
        self.data.alpha_matte_smoothed = smoothing.smoothing(self.data)
        return self.data.alpha_matte_smoothed
