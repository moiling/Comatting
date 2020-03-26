#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/3/26 15:05
# @Author  : moiling
# @File    : color_space_matting.py

import time

from .color_space import ColorSpace
from ..data import MattingData
import numpy as np


class ColorSpaceMatting:
    def __init__(self, data: MattingData):
        self.data = data

    def matting(self, max_fes):
        # color space.
        color_space = ColorSpace(self.data)
