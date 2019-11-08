# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 09:47:14 2019

@author: Michael
"""

import cv2
import numpy as np
import basic_functions as bf

def gaussian_kernel(sigma, r):
    
    size = 2 * r + 1
    
    kernel = np.zeros([size, size])
    
    k = (2 * sigma**2)
    k_pi = k / np.pi
    
    for i in range(0, size):
        for j in range(0, size):
            
            kernel[i][j] = k_pi * np.exp( -( (i - r)**2 + (j - r)**2 ) / k )
    
    return np.array(np.round(kernel / kernel[0][0]), np.int)


# eXtended Difference of Gaussian
def XDOG(img, k, sigma, p):
    
    img_temp = img.copy()
    
#    Check for RGB image, if so then convert to grayscale
    if img_temp.ndim == 3:
        img_temp = cv2.cvtColor(img_temp, cv2.COLOR_BGR2GRAY)
    
    img_temp = np.array(img_temp, np.double)
    
    gauss_1 = gaussian_kernel(sigma, 2)
    gauss_2 = gaussian_kernel(k * sigma, 2)
    
    gauss_1_sum = np.sum(gauss_1)
    gauss_2_sum = np.sum(gauss_2)
    
    G1 = bf.spatial_filter(img_temp, gauss_1 / gauss_1_sum)
    G2 = bf.spatial_filter(img_temp, gauss_2 / gauss_2_sum)
    
    img_temp[:,:] = (1 + p) * G1[:,:] - p * G2[:,:]
    
    return np.array(img_temp, np.uint8)

def simple_threshold(img, cutoff):
    
    out = np.zeros_like(img)
    
    for i in range(0, len(img)):
        for j in range(0, len(img[i])):
            
            if img[i][j] > cutoff:
                out[i][j] = 1
    
    return out
    
def better_thereshold(img, cutoff, phi):
    
    out = np.ones_like(img)
    
    for i in range(0, len(img)):
        for j in range(0, len(img[i])):
            
            if img[i][j] <= cutoff:
                out[i][j] = 1 + np.tanh(phi * (img[i][j]) - cutoff)
            
    return out