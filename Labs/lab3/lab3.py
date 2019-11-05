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
out_img3 = fun.adaptive_histogram(img1, 16, 16)
out_img4 = fun.adaptive_histogram(img2, 16, 16)

bf.display_multiple_images([img1, out_img3])
bf.display_multiple_images([img2, out_img4])

# Question 3 and 4

img3 = cv2.imread(r"Images\testing\sharpen\7.2.01-small.png", 0)
img4 = cv2.imread(r"Images\testing\sharpen\digital_orca_blurred.png", 0)

r = 2
k = 1

# Unsharpen Mask

out_img5 = fun.unsharp_mask(img3, r, k)
out_img6 = fun.unsharp_mask(img4, r, k)

bf.display_multiple_images([img3, out_img5])
bf.display_multiple_images([img4, out_img6])

# Laplacian Sharpen

k = 1

out_img7 = fun.laplacian_sharpen(img3, k)
out_img8 = fun.laplacian_sharpen(img4, k)

bf.display_multiple_images([img3, out_img7])
bf.display_multiple_images([img4, out_img8])


# Section 2.3

img5 = cv2.imread(r"Images\enhance\noise_additive.png", 0)
img6 = cv2.imread(r"Images\enhance\noise_multiplicative.png", 0)
img7 = cv2.imread(r"Images\enhance\noise_impulsive.png", 0)
img8 = cv2.imread(r"Images\enhance\snowglobe.png", 0)

# Additive Noise

out_img9 = cv2.medianBlur(img5, 3)

k = 1
r = 3
out_img9 = fun.unsharp_mask(out_img9, r, k)

k = 0.4
out_img9 = fun.laplacian_sharpen(out_img9, k)

bf.display_multiple_images([img5, out_img9])

# Multiplicative Nosie

average = np.array([[1, 1, 1],
                    [1, 1, 1],
                    [1, 1, 1]])

out_img10 = np.array(bf.spatial_filter(img6, average / np.sum(average)), np.uint8)

k = 0.1
out_img10 = fun.laplacian_sharpen(out_img10, k)

bf.display_multiple_images([img6, out_img10])

# Impulsive Noise

out_img11 = cv2.medianBlur(img7, 3)

bf.display_multiple_images([img7, out_img11])

# Snowglobe

r = 2
gauss = fun.gaussian_kernel(1, r)
gauss_sum = np.sum(gauss)

out_img12 = np.array(bf.spatial_filter(img8, gauss / gauss_sum), np.uint8)
out_img12 = fun.histogram_equalization(out_img12)

bf.display_multiple_images([img8, out_img12])