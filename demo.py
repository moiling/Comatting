#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/3/23 12:40
# @Author  : moiling
# @File    : demo.py
import os
import time
import cv2
import numpy as np
from matting import Matting

root_path = '/Users/moi/Documents/Code/PycharmProjects/ColorSpaceMatting/data/'
version_name, resize_name = 'v0.1', '/'

if __name__ == '__main__':

    # for img_name in ['elephant', 'donkey', 'doll', 'net', 'pineapple', 'plant', 'plasticbag', 'troll']:
    for img_name in ['GT14']:
        img_url = root_path + 'input_lowres/' + resize_name + img_name + '.png'
        trimap_url = root_path + 'trimap_lowres/Trimap1/' + resize_name + img_name + '.png'
        out_url = './out/'

        matting = Matting(img_url, trimap_url, img_name)

        time_start = time.time()
        matting.comatting()
        print('img:%s, time used:%.4f s' % (img_name, time.time() - time_start))

        img_f, img_b = matting.img_fnb()

        save_path = out_url + version_name + '/' + resize_name
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        cv2.imwrite(save_path + img_name + '_f.png', img_f)
        cv2.imwrite(save_path + img_name + '_b.png', img_b)
        cv2.imwrite(save_path + img_name + '_alpha.png', matting.data.alpha_matte)
        np.save(save_path + img_name + '_f', matting.data.sample_f)
        np.save(save_path + img_name + '_b', matting.data.sample_b)
