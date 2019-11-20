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

out_img1 = fun.rgb_to_hsi(img1)

out_img2 = fun.rgb_to_hsi(out_img1)

bf.display_multiple_images([img1, out_img2])