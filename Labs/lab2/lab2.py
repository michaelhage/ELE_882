# -*- coding: utf-8 -*-
"""
Created on Sun Oct 13 17:44:50 2019

@author: Michael
"""

import sys
sys.path.append(r"B:\School\Fourth_Year\ELE_882\Labs\lab2")
import numpy as np
import lab2_functions as fun
import basic_functions as bf
import cv2

img1 = cv2.imread(r"Images\nms-test.png", 0)
img2 = cv2.imread(r"Images\threshold-test.png", 0)
img3 = cv2.imread(r"Images\Lena-gray.tif", 0)

img1_double = np.array(img1, np.double)
img2_double = np.array(img2, np.double)
img3_double = np.array(img3, np.double)

# section 2.1.1

# Question 5

# Spatial Filter Test

# create gaussian kernel, 5x5

g_kernel = [[1, 4, 7, 4, 1],
            [4, 20, 33, 20, 4],
            [7, 33, 55, 33, 7],
            [4, 20, 33, 20, 4],
            [1, 4, 7, 4, 1]]

g_sum = np.sum(g_kernel)

g_kernel = g_kernel / g_sum

out_img = np.array(fun.spatial_filter(img3_double, g_kernel), np.uint8)

#bf.display_multiple_images([img3, out_img])

# NMS filter Test

nms_out_img_h, nms_out_img_v = np.array(fun.non_max_suppress(img1_double, 5, 5), np.uint8)

nms_out_img = np.zeros_like(nms_out_img_h)
for i in range(0,len(nms_out_img_h)):
    for j in range(0,len(nms_out_img_h[i])):
        if nms_out_img_h[i][j] == nms_out_img_v[i][j]:
            nms_out_img[i][j] = nms_out_img_h[i][j]

# Threshold Filter Test