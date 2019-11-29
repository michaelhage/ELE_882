# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 09:14:44 2019

@author: Michael
"""

import numpy as np
import basic_functions as bf
import lab5_functions as fun
import cv2

img1 = cv2.imread(r"Images/Lenna.png", 1)
img2 = cv2.imread(r"Images/4.1.02.tiff", 1)
img3 = cv2.imread(r"Images/mandrill.png", 1)
img4 = cv2.imread(r"Images/snowglobe_colour.png", 1)
img5 = cv2.imread(r"Images/pep.png", 1)

# Convert RGB -> HSI -> RGB
out_img1 = fun.rgb_to_hsi(img5)

out_img2 = fun.hsi_to_rgb(out_img1)

bf.display_multiple_images([img5, out_img2])

# Convert RGB -> YCbCr -> RGB
out_img3 = fun.rgb_to_ycbcr(img1)

out_img4 = fun.ycbcr_to_rgb(out_img3)

bf.display_multiple_images([img1, out_img4])

# Rotate Hue

out_img5 = fun.change_hue(img5, np.pi / 6)

bf.display_multiple_images([img5, out_img5])

# Offset Saturation

out_img6 = fun.change_saturation(img5, -0.1)

bf.display_multiple_images([img5, out_img6])

# Point Transform Colour
out_img7 = fun.apply_point_tfrm(img5, 2, -20)

bf.display_multiple_images([img5, out_img7])

# Colour Spatial Filter

# Gaussian Kernel
sigma = 1
r = 2

gauss, gauss_sum = fun.gaussian_kernel(sigma, r)
out_img8 = fun.spatial_filter(img1, gauss / gauss_sum)

bf.display_multiple_images([img1, out_img8])

# Histogram Equalize

out_img9 = fun.histogram_equalization_rgb(img1)

bf.display_multiple_images([img1, out_img9])

bf.display_color_histogram(img1)
bf.display_color_histogram(out_img9)

# Histogram Equalize YCbCr
out_img10 = fun.histogram_equalization_ycbcr(img1)

bf.display_multiple_images([img1, out_img10])

bf.display_color_histogram(img1)
bf.display_color_histogram(out_img10)