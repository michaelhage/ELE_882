# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 20:32:55 2019

@author: Michael
"""

import numpy as np
import basic_functions as bf

# Create Gaussian Kernel Size r
def gaussian_kernel(sigma, r):
    
    size = 2 * r + 1
    
    kernel = np.zeros([size, size])
    
    m, m = bf.padding_param(size)
    
    k = (2 * sigma**2)
    k_pi = k / np.pi
    
    for i in range(0, size):
        for j in range(0, size):
            
            kernel[i][j] = k_pi * np.exp( -( (i - m)**2 + (j - m)**2 ) / k )
    
    return np.array(np.round(kernel / kernel[0][0]), np.int)

#Apply Global Histogram Equalization
def histogram_equalization(img):
    
#    Check for unsigned integer 8-bit
    if img.dtype != np.uint8:
        return img
    
    out_img = img.copy()
    MAX = np.iinfo(np.uint8).max
    
#    Creates the histogram
    his = np.zeros(MAX+1)
    
#    Calculate the histogram
    for i in range(0, len(img)):
        for j in range(0, len(img[i])):
            
            his[img[i][j]] += 1
     
#    Calculate the CDF
    cdf = np.zeros_like(his)    
    cdf[0] = his[0]
    
    for i in range(1,len(cdf)):
        cdf[i] = his[i] + cdf[i - 1]
    
    cdf = cdf / img.size * MAX
    
#    Apply the CDF
    for i in range(0, len(img)):
        for j in range(0, len(img[i])):
            
            out_img[i][j] = cdf[img[i][j]]
    
    return out_img

#Apply Local Histogram Equalization
def adaptive_histogram(img, H, W):
    
#    Check for unsigned integer 8-bit
    if img.dtype != np.uint8:
        return img
    
    out_img = img.copy()
    MAX = np.iinfo(np.uint8).max
    m,n = img.shape
    
    pdf = np.zeros(MAX+1)
    cdf = np.zeros(MAX+1)
    
#    Zero padding the Image
    col_right, col_left = bf.padding_param(W)
    row_bottom, row_top = bf.padding_param(H)
    
    img_temp = np.array(np.zeros([m+row_top+row_bottom,n+col_left+col_right]), np.uint8)
    img_temp[row_top:m+row_top, col_left:n+col_left] = 1
    img_temp[row_top:m+row_top, col_left:n+col_left] *= img
    
#    Iterate over sections and perform the local histogram equalize operation
    for i in range(row_top,m+row_top):
        for j in range(col_left,n+col_left):
            
#            Reset All Parameters
            cdf[:] = 0
            pdf[:] = 0
            
            snap = img_temp[i-row_top: i+row_bottom+1, j-col_left: j+col_right+1].copy()
            for x in range(0,len(snap)):
                for y in range(0,len(snap[x])):
                    
                    pdf[snap[x][y]] += 1
        
            cdf[0] = pdf[0]
            
            for a in range(1,MAX+1):
                cdf[a] = pdf[a] + cdf[a - 1]
            
            out_img[i-row_top][j-col_left] = cdf[snap[row_top][col_left]] * MAX / snap.size
            
    return out_img

# Unsharpen Image
def unsharp_mask(img, r, k):
    
    if img.dtype != np.uint8:
        return img
    
    out_img = np.array(img.copy(), np.double)
    img_temp = img.copy()
    
#    Make gaussian kernel
    gauss = gaussian_kernel(1, r)
    gauss_sum = np.sum(gauss)
    
#    Apply gaussian kernel
    img_blur = bf.spatial_filter(img, gauss / gauss_sum)
    
#    Perform unsharp operation
    out_img = img_temp - k * (img_temp - img_blur)
    
    return np.array(out_img, np.uint8)

#Sharpen an Image using Laplacian Kernel
def laplacian_sharpen(img, k):
    
    out_img = np.array(img.copy(), np.double)
    img_temp = img.copy()
    
#    Laplace kernel
    laplace = np.array([[-1, -1, -1],
                        [-1, 8, -1],
                        [-1, -1, -1]])
    
#    Apply kernel and operation
    img_lpc = bf.spatial_filter(img, laplace)
    out_img = img_temp + k * img_lpc
    
    for i in range(0, len(out_img)):
        for j in range(0, len(out_img[i])):
            
            if out_img[i][j] < 0:
                out_img[i][j] = 0
            elif out_img[i][j] > 255:
                out_img[i][j] = 255
    
    return np.array(out_img, np.uint8)
