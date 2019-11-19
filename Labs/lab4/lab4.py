# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 09:41:47 2019

@author: Michael
"""

import cv2
import numpy as np
import lab4_functions as fun
import basic_functions as bf

img1 = cv2.imread(r"Images/Lena-gray.tif", 0)
img2 = cv2.imread(r"Images/Lenna.png", 1)
img3 = cv2.imread(r"Images/mandrill.tif", 0)
img4 = cv2.imread(r"Images/mandrill.png", 1)

# Section 3.1

# Section 3.1.1

# Problem 1
k = 2
sigma = 1
p = 0.5

out_img1 = fun.XDOG(img1, k, sigma, p)
out_img2 = fun.XDOG(img2, k, sigma, p)

bf.display_multiple_images([img1, out_img1])
bf.display_multiple_images([img2, out_img2])

# Problem 2

cutoff = 150
phi = 1.5

# hard Thresholding
out_img3 = fun.hard_threshold(out_img1, cutoff)
bf.display_multiple_images([img1, out_img3])

#Soft Thresholding
out_img4 = fun.soft_thereshold(out_img1, cutoff, phi)
bf.display_multiple_images([img1, out_img4])


# Problem 3
cutoff = 150
phi = 0.5

out_img5 = fun.three_tone(out_img1, cutoff, phi)
bf.display_multiple_images([img1, out_img1, out_img5])


# Section 3.2

R = 8
gamma = 2

out_img6 = fun.oilify(img1, R, gamma)
bf.display_multiple_images([img1, out_img6])

# Section 3.3

# Problem 1

min_window_size = 1
iteration = 3

out_img7 = fun.edge_preserving(img1, min_window_size, iteration)  
bf.display_multiple_images([img1, out_img7])

# Problem 2
# Edge Detector
sigma = 0.33

out_img8 = fun.extract_edges(img1, sigma = sigma)
bf.display_multiple_images([img1, out_img8])

# Problem 3
# Implement the Cartoon Effect

min_window_size = 3
iteration = 3
sigma = 0.33

out_img9 = fun.cartoon_effect(img1, min_window_size, iteration, sigma = sigma)
bf.display_multiple_images([img1, out_img9])

# Problem 4
# Incorporates the XDOG function with the cartoon effect

min_window_size = 3
iteration = 3
sigma = 0.33
k = 3
p = 5
sigma_x = 1

out_img10 = fun.cartoon_effect(img1, min_window_size, iteration, sigma = sigma, k = k, p = p, sigma_x = sigma_x, flag = 1)
bf.display_multiple_images([img1, out_img10])
