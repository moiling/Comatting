#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/3/23 13:12
# @Author  : moiling
# @File    : matting.py
from enum import Enum

import cv2

from .color_closely_segmentation.color_closely_segmentation import ColorCloselySegmentation
from .color_space_matting.color_space_matting import ColorSpaceMatting, EvolutionType
from .multi_points_matting.multi_points_matting import EvolutionType as MultiEvolutionType
from .loss import sad_loss, mse_loss
from .multi_points_matting.multi_points_matting import MultiPointsMatting
from .random_matting.random_matting import RandomMatting
from .smoothing import smoothing
from .comatting.comatting import Comatting
from .data import MattingData
from .spatial_matting.spatial_matting import SpatialMatting
from .vanilla_matting.vanilla_matting import VanillaMatting


class Method(Enum):
    COMATTING = 'comatting'
    RANDOM = 'random_matting'
    RANDOM_B_RAY = 'random_matting(ray_b)'
    COLOR_SPACE = 'color_space_matting'
    SPATIAL = 'spatial_matting'
    VANILLA = 'vanilla_matting'
    SPATIAL_UC = 'spatial_matting(uc)'
    RANDOM_UC = 'random_matting(uc)'
    COLOR_SPACE_B_RAY_SAMPLE = 'color_space(ray_b_sample)'
    COLOR_SPACE_B_RAY = 'color_space(ray_b)'
    MULTI_POINTS = 'multi_points_matting'
    MULTI_POINTS_VANILLA = 'multi_vanilla_matting'
    MULTI_POINTS_FB = 'multi_points_matting(FB)'
    MULTI_RANDOM = 'multi_random_matting'
    MULTI_RANDOM_FB = 'multi_random_matting(FB)'
    MULTI_RANDOM_SINGLE = 'multi_random_matting(single)'


class Matting:
    default_max_fes = 1e3

    def __init__(self, img_url, trimap_url, img_name='', log=False):
        self.data = MattingData(img_url, trimap_url, img_name, log)

    def matting(self, func_name=Method.COMATTING, max_fes=default_max_fes):
        if func_name == Method.COMATTING:
            return self.comatting(max_fes)
        if func_name == Method.RANDOM:
            return self.random_matting(max_fes)
        if func_name == Method.COLOR_SPACE:
            return self.color_space_matting(max_fes)
        if func_name == Method.SPATIAL:
            return self.spatial_matting(max_fes)
        if func_name == Method.VANILLA:
            return self.vanilla_matting(max_fes)
        if func_name == Method.SPATIAL_UC:
            return self.spatial_matting(max_fes, unique_color=True)
        if func_name == Method.RANDOM_UC:
            return self.random_matting(max_fes, unique_color=True)
        if func_name == Method.COLOR_SPACE_B_RAY_SAMPLE:
            return self.color_space_matting(max_fes, EvolutionType.B_RAY_SAMPLE)
        if func_name == Method.COLOR_SPACE_B_RAY:
            return self.color_space_matting(max_fes, EvolutionType.B_RAY)
        if func_name == Method.RANDOM_B_RAY:
            return self.color_space_matting(max_fes, EvolutionType.B_RAY_RANDOM)
        if func_name == Method.MULTI_POINTS:
            return self.multi_points_matting(max_fes)
        if func_name == Method.MULTI_POINTS_VANILLA:
            return self.multi_points_matting(max_fes, MultiEvolutionType.VANILLA)
        if func_name == Method.MULTI_POINTS_FB:
            return self.multi_points_matting(max_fes, MultiEvolutionType.MIN_FB_RGB)
        if func_name == Method.MULTI_RANDOM:
            return self.multi_points_matting(max_fes, MultiEvolutionType.RANDOM_MIN_FB_RGB_ALPHA)
        if func_name == Method.MULTI_RANDOM_FB:
            return self.multi_points_matting(max_fes, MultiEvolutionType.RANDOM_FB_RGB)
        if func_name == Method.MULTI_RANDOM_SINGLE:
            return self.multi_points_matting(max_fes, MultiEvolutionType.RANDOM_SINGLE)

        raise Exception('ERROR: no matting function named {}.'.format(func_name))

    def multi_points_matting(self, max_fes=default_max_fes, evo_type=MultiEvolutionType.MIN_FB_RGB_ALPHA):
        MultiPointsMatting(self.data).matting(max_fes, evo_type)
        return self.data.alpha_matte

    def vanilla_matting(self, max_fes=default_max_fes):
        VanillaMatting(self.data).matting(max_fes)
        return self.data.alpha_matte

    def spatial_matting(self, max_fes=default_max_fes, unique_color=False):
        SpatialMatting(self.data).matting(max_fes, unique_color)
        return self.data.alpha_matte

    def color_closely_segmentation(self):
        return ColorCloselySegmentation(self.data).segment()

    def comatting(self, max_fes=default_max_fes):
        Comatting(self.data).matting(max_fes)
        return self.data.alpha_matte

    def color_space_matting(self, max_fes=default_max_fes, evolution_type: EvolutionType = EvolutionType.VANILLA):
        ColorSpaceMatting(self.data).matting(max_fes, evolution_type)
        return self.data.alpha_matte

    def random_matting(self, max_fes=default_max_fes, unique_color=False):
        RandomMatting(self.data).matting(max_fes, unique_color)
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

    def clear_result(self):
        self.data.clear_result()

    def loss(self, gt_url):
        gt = cv2.imread(gt_url, cv2.IMREAD_GRAYSCALE)
        sad = sad_loss(self.data, gt)
        mse = mse_loss(self.data, gt)

        return sad, mse
