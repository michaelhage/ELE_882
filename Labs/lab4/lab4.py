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

k = 2
sigma = 1
p = 0.5

out_img1 = fun.XDOG(img1, k, sigma, p)
out_img2 = fun.XDOG(img2, k, sigma, p)

bf.display_multiple_images([img1, out_img1])
bf.display_multiple_images([img2, out_img2])