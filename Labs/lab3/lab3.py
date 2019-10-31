# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 20:30:42 2019

@author: Michael
"""

import numpy as np
import lab3_functions as fun
import basic_functions as bf
import cv2

# Section 2.2 Testing

# Question 1 and 2

# Histogram Equalize

img1 = cv2.imread(r"Images\testing\contrast\7.2.01-small.png", 0)
img2 = cv2.imread(r"Images\testing\contrast\207056.jpg", 0)

# Global Equalize
out_img1 = fun.histogram_equalization(img1)
out_img2 = fun.histogram_equalization(img2)

bf.display_multiple_images([img1, out_img1])
bf.display_multiple_images([img2, out_img2])

# Adaptive Histogram Equalize
out_img3 = fun.adaptive_histogram(img1, 7, 7)
out_img4 = fun.adaptive_histogram(img2, 7, 7)

bf.display_multiple_images([img1, out_img3])
bf.display_multiple_images([img2, out_img4])

# Question 3 and 4

img3 = cv2.imread(r"Images\testing\sharpen\7.2.01-small.png", 0)
img4 = cv2.imread(r"Images\testing\sharpen\digital_orca_blurred.png", 0)

r = 2
k = 0.2

# Unsharpen Mask

out_img5 = fun.unsharp_mask(img3, r, k)
out_img6 = fun.unsharp_mask(img4, r, k)

bf.display_multiple_images([img3, out_img5])
bf.display_multiple_images([img4, out_img6])

# Laplacian Sharpen

out_img7 = fun.laplacian_sharpen(img3, k)
out_img8 = fun.laplacian_sharpen(img4, k)

bf.display_multiple_images([img3, out_img7])
bf.display_multiple_images([img4, out_img8])

# CLAHE 

