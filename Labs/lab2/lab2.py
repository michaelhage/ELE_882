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
img4 = cv2.imread(r"Images\mandrill.tif", 0)

# Section 2.1 - Filtering Functions

# Question 4

# Spatial Filter Test

# create gaussian kernel, 5x5

g_kernel = np.array([[1, 4, 7, 4, 1],
            [4, 20, 33, 20, 4],
            [7, 33, 55, 33, 7],
            [4, 20, 33, 20, 4],
            [1, 4, 7, 4, 1]])

g_sum = np.sum(g_kernel)

g_kernel = g_kernel / g_sum

out_img = fun.spatial_filter(img3, g_kernel)

#display
#bf.display_multiple_images([img3, out_img])

# NMS filter Test

nms_out_img = fun.non_max_suppress_full(img1, 5, 5)


#display
#bf.display_multiple_images([img1, nms_out_img])

# Threshold Filter Test

# Set thresholding value
T = 0.25         

thresh_out_img = fun.image_thresholding(img2, T)

#display
#bf.display_multiple_images([img2, thresh_out_img])

# Section 2.2 - Edge Detector

H = np.array([[1, 0, -1],
              [2, 0, -2],
              [1, 0, -1]])