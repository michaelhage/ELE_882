# -*- coding: utf-8 -*-
"""
Created on Sun Oct 13 17:44:50 2019

@author: Michael
"""

import numpy as np
import lab2_functions as fun
import basic_functions as bf
import cv2

img1 = cv2.imread(r"Images\nms-test.png", 0)
img2 = cv2.imread(r"Images\threshold-test.png", 0)
img3 = cv2.imread(r"Images\Lena-gray.tif", 0)
img4 = cv2.imread(r"Images\mandrill.tif", 0)

# Section 2.1 - Filtering Functions

# Spatial Filter Test

# create gaussian kernel, 5x5

g_kernel = np.array([[1, 4, 7, 4, 1],
            [4, 20, 33, 20, 4],
            [7, 33, 55, 33, 7],
            [4, 20, 33, 20, 4],
            [1, 4, 7, 4, 1]])

g_sum = np.sum(g_kernel)

g_kernel = g_kernel / g_sum

out_img = np.array(fun.spatial_filter(img3, g_kernel), np.uint8)

#display
bf.display_multiple_images([img3, out_img])


# NMS filter Test

#This NMS filter uses a 2D kernel
nms_out_img = fun.non_max_suppress_full(img1, 5, 5)

#This NMS filter uses two vector kernels (one horizontal and one vertical) 
#and produces two output imaegs
nms_out_img_x, nms_out_img_y = fun.non_max_suppress(img1, 5, 5)

#display
bf.display_multiple_images([img1, nms_out_img, nms_out_img_x, nms_out_img_y])

# Threshold Filter Test

# Set thresholding value
T = 0.25         

thresh_out_img = fun.image_thresholding(img2, T)

#display
bf.display_multiple_images([img2, thresh_out_img])


# Section 2.2 - Edge Detector

#help(fun.edge_detector)

#defining parameters
H = np.array([[1, 0, -1],
              [2, 0, -2],
              [1, 0, -1]])

#both perfom the same operation, but one of them uses the default parameters 
#and the other can be adjusted

#default parameters
out_img_edge = fun.edge_detector(img3, H)

#display
bf.display_multiple_images([img3, out_img_edge])

#manual parameters
threshold = 0.3
size = 7

out_img_edge = fun.edge_detector(img3, H, T = threshold, wndsz = size)

bf.display_multiple_images([img3, out_img_edge])


#Section 2.3 - Derivative Filters


#use the following command to see the help function
#help(fun.derivative_kernel)

#This variable will select the kernel
#The spatial filter doesn't work properly with the vector inputs
sel = 2

#calls the kernel selection function
H = fun.derivative_kernel(sel)

out_img_edge = fun.edge_detector(img3, H)

#display
bf.display_multiple_images([img3, out_img_edge])

