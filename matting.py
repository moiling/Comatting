#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/3/23 13:12
# @Author  : moiling
# @File    : matting.py
from comatting import Comatting
from data import MattingData


class Matting:
    def __init__(self, img_url, trimap_url, img_name=''):
        self.data = MattingData(img_url, trimap_url, img_name)

    def matting(self, func_name='comatting'):
        if func_name == 'comatting':
            Comatting(self.data).matting()
        else:
            raise Exception('ERROR: no matting function named {}.'.format(func_name))

    def comatting(self):
        Comatting(self.data).matting()
        return self.data.alpha_matte

    def img_fnb(self):
        self.data.img_fnb()
        return self.data.img_f, self.data.img_b

