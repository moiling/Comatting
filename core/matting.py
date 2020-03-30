#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/3/23 13:12
# @Author  : moiling
# @File    : matting.py
from .color_closely_segmentation.color_closely_segmentation import ColorCloselySegmentation
from .color_space_matting.color_space_matting import ColorSpaceMatting
from .random_matting.random_matting import RandomMatting
from .smoothing import smoothing
from .comatting.comatting import Comatting
from .data import MattingData
from .spatial_matting.spatial_matting import SpatialMatting
from .vanilla_matting.vanilla_matting import VanillaMatting


class Matting:
    default_max_fes = 1e3

    def __init__(self, img_url, trimap_url, img_name=''):
        self.data = MattingData(img_url, trimap_url, img_name)

    def matting(self, func_name='comatting', max_fes=default_max_fes):
        if func_name == 'comatting':
            return self.comatting(max_fes)
        if func_name == 'random_matting':
            return self.random_matting(max_fes)
        if func_name == 'color_space_matting':
            return self.color_space_matting(max_fes)
        if func_name == 'spatial_matting':
            return self.spatial_matting(max_fes)
        if func_name == 'vanilla_matting':
            return self.vanilla_matting(max_fes)

        raise Exception('ERROR: no matting function named {}.'.format(func_name))

    def vanilla_matting(self, max_fes=default_max_fes):
        VanillaMatting(self.data).matting(max_fes)
        return self.data.alpha_matte

    def spatial_matting(self, max_fes=default_max_fes):
        SpatialMatting(self.data).matting(max_fes)
        return self.data.alpha_matte

    def color_closely_segmentation(self):
        return ColorCloselySegmentation(self.data).segment()

    def comatting(self, max_fes=default_max_fes):
        Comatting(self.data).matting(max_fes)
        return self.data.alpha_matte

    def color_space_matting(self, max_fes=default_max_fes):
        ColorSpaceMatting(self.data).matting(max_fes)
        return self.data.alpha_matte

    def random_matting(self, max_fes=default_max_fes):
        RandomMatting(self.data).matting(max_fes)
        return self.data.alpha_matte

    def img_fnb(self):
        self.data.img_fnb()
        return self.data.img_f, self.data.img_b

    def smoothing(self):
        self.data.alpha_matte_smoothed = smoothing.smoothing(self.data)
        return self.data.alpha_matte_smoothed

    def outline(self, img, width=2, f_line_color=None, b_line_color=None):
        if b_line_color is None:
            b_line_color = [0, 255, 255]
        if f_line_color is None:
            f_line_color = [255, 255, 0]
        return self.data.add_outline(img, width, f_line_color, b_line_color)
