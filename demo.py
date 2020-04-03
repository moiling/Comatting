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
version_name, resize_name = 'v0.3.1', ''
func_names = ['random_matting', 'random_matting(uc)', 'comatting', 'color_space_matting', 'spatial_matting(uc)',
              'spatial_matting', 'vanilla_matting']
# func_names = ['spatial_matting(uc)']

max_fes_list = [1e1, 5e1, 1e3, 5e3]
# max_fes_list = [5e1]
log_in_method = False

if __name__ == '__main__':

    for max_fes in max_fes_list:
        # Log
        print('{:-^161}\n|{:^159}|\n|{:^159}|\n{:-^161}\n|'
              '{:^15}|{:^10}|{:^15}|{:^25}|{:^10}|{:^15}|{:^15}|{:^15}|{:^15}|{:^15}|\n{:-^161}'
              .format('', 'MATTING', time.strftime('%b %d %Y %H:%M:%S', time.localtime(time.time())), '', 'START TIME',
                      'IMAGE', 'SIZE', 'METHOD', 'FES', 'AVG FIT', 'SAD', 'MSE', 'MATTING TIME', 'SMOOTHING TIME', ''))

        # for img_name in ['elephant', 'donkey', 'doll', 'net', 'pineapple', 'plant', 'plasticbag', 'troll']:
        # for img_name in ['GT14']:
        # for img_name in ['GT{:02d}'.format(i) for i in range(1, 28)]:
        for img_name in ['GT{:02d}'.format(i) for i in [1, 4, 5, 13, 14, 16, 18, 21, 24, 25, 27]]:  # selected img
            img_url = '{}/input_lowres/{}/{}.png'.format(root_path, resize_name, img_name)
            trimap_url = '{}/trimap_lowres/Trimap1/{}/{}.png'.format(root_path, resize_name, img_name)
            gt_url = '{}/gt/{}/{}.png'.format(root_path, resize_name, img_name)
            out_url = './out'
            matting = Matting(img_url, trimap_url, img_name, log=log_in_method)

            for func_name in func_names:
                # Log
                print('|{:^15}|{:^10}|{:^15}|{:^25}|{:^10.0e}'
                      .format(time.strftime('%H:%M:%S', time.localtime(time.time())), img_name,
                              '{}x{}'.format(matting.data.height, matting.data.width), func_name, max_fes), end='')

                time_start = time.time()
                alpha_matte = matting.matting(func_name=func_name, max_fes=max_fes)
                matting_time = time.time() - time_start
                time_start = time.time()
                alpha_matte_smoothed = matting.smoothing()
                smoothing_time = time.time() - time_start

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
                np.save('{}/{}_fit'.format(save_data_path, img_name), matting.data.fit)

                # outline
                outline_f = matting.outline(img_f)
                outline_b = matting.outline(img_b)
                cv2.imwrite('{}/{}_outline_f.png'.format(save_pic_path, img_name), outline_f)
                cv2.imwrite('{}/{}_outline_b.png'.format(save_pic_path, img_name), outline_b)

                sad_loss, mse_loss = np.nan, np.nan
                # error with ground truth
                if os.path.exists(gt_url):
                    sad_loss, mse_loss = matting.loss(gt_url)
                    np.save('{}/{}_sad'.format(save_data_path, img_name), sad_loss)
                    np.save('{}/{}_mse'.format(save_data_path, img_name), mse_loss)

                # Log
                print('|{:^15.3f}|{:^15.3f}|{:^15.3f}|{:^15.2f}|{:^15.2f}|\n{:-^161}'
                      .format(np.sum(matting.data.fit) / matting.data.u_size, sad_loss, mse_loss, matting_time,
                              smoothing_time, ''))

                # clear to avoid wrong result.
                matting.clear_result()
