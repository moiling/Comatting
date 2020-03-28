#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/3/23 12:40
# @Author  : moiling
# @File    : demo.py
import os
import time
import cv2
import numpy as np
from core.matting import Matting

root_path = '/Users/moi/Documents/Code/PycharmProjects/ColorSpaceMatting/data'
version_name, resize_name = 'v0.3', ''
func_name = 'color_space_matting'
max_fes = 1e3

if __name__ == '__main__':

    for img_name in ['elephant', 'donkey', 'doll', 'net', 'pineapple', 'plant', 'plasticbag', 'troll']:
        # for img_name in ['GT14']:
        img_url = '{}/input_lowres/{}/{}.png'.format(root_path, resize_name, img_name)
        trimap_url = '{}/trimap_lowres/Trimap1/{}/{}.png'.format(root_path, resize_name, img_name)
        out_url = './out'

        matting = Matting(img_url, trimap_url, img_name)

        time_start = time.time()
        alpha_matte = matting.matting(func_name=func_name, max_fes=max_fes)
        print('img:{}, matting time used:{:.4f}s'.format(img_name, time.time() - time_start))

        time_start = time.time()
        alpha_matte_smoothed = matting.smoothing()
        print('img:{}, smoothing time used:{:.4f}s'.format(img_name, time.time() - time_start))

        img_f, img_b = matting.img_fnb()

        save_pic_path = '{}/{}/{}/{:.0e}/{}'.format(out_url, version_name, func_name, max_fes, resize_name)
        save_data_path = '{}/data/'.format(save_pic_path)
        if not os.path.exists(save_pic_path):
            os.makedirs(save_pic_path)
        if not os.path.exists(save_data_path):
            os.makedirs(save_data_path)

        cv2.imwrite('{}/{}_f.png'.format(save_pic_path, img_name), img_f)
        cv2.imwrite('{}/{}_b.png'.format(save_pic_path, img_name), img_b)
        cv2.imwrite('{}/{}_alpha.png'.format(save_pic_path, img_name), alpha_matte)
        cv2.imwrite('{}/{}_alpha_smoothed.png'.format(save_pic_path, img_name), alpha_matte_smoothed)
        np.save('{}/{}_f'.format(save_data_path, img_name), matting.data.sample_f)
        np.save('{}/{}_b'.format(save_data_path, img_name), matting.data.sample_b)
        np.save('{}/{}_cost_c'.format(save_data_path, img_name), matting.data.cost_c)

        # outline
        outline_f = matting.outline(img_f)
        outline_b = matting.outline(img_b)
        cv2.imwrite('{}/{}_outline_f.png'.format(save_pic_path, img_name), outline_f)
        cv2.imwrite('{}/{}_outline_b.png'.format(save_pic_path, img_name), outline_b)
