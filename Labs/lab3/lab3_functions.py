# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 20:32:55 2019

@author: Michael
"""

import numpy as np
import cv2
import basic_functions as bf

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

def laplace_gaussian_kernel(sigma, r):
    
    size = 2 * r + 1
    
    kernel = np.zeros([size, size])
    
    m, m = bf.padding_param(size)
    
    k_2 = 2 * sigma**2
    k_6 = 2 * np.pi * sigma**6
    
    for i in range(0, size):
        for j in range(0, size):
            
            kernel[i][j] = ( (i**2 + j**2 - k_2) / k_6) * np.exp( -( (i - m)**2 + (j - m)**2 ) / k_2 )
    
    return np.array(np.round(kernel / kernel[0][0]), np.int)

def histogram_equalization(img):
    
    if img.dtype != np.uint8:
        return img
    
    out_img = img.copy()
    MAX = np.iinfo(np.uint8).max
    
    his = np.zeros(MAX)
    
    for i in range(0, len(img)):
        for j in range(0, len(img[i])):
            
            his[img[i][j]] += 1
            
    cdf = np.zeros_like(his)    
    cdf[0] = his[0]
    
    for i in range(1,len(cdf)):
        cdf[i] = his[i] + cdf[i - 1]
    
    cdf = cdf / img.size * MAX
    
    for i in range(0, len(img)):
        for j in range(0, len(img[i])):
            
            out_img[i][j] = cdf[img[i][j]]
    
    return out_img


def adaptive_histogram(img, H, W):
    
     
    if img.dtype != np.uint8:
        return img
    
    out_img = img.copy()
    MAX = np.iinfo(np.uint8).max
    m,n = img.shape
    
    pdf = np.zeros(MAX)
    his = np.zeros(MAX)
    cdf = np.zeros(MAX)
    
    col_right, col_left = bf.padding_param(W)
    row_bottom, row_top = bf.padding_param(H)
    
    img_temp = np.array(np.zeros([m+row_top+row_bottom,n+col_left+col_right]), np.uint8)
    img_temp[row_top:m+row_top, col_left:n+col_left] = 1
    img_temp[row_top:m+row_top, col_left:n+col_left] *= img

    for i in range(row_top,m+row_top):
        for j in range(col_left,n+col_left):
            
            cdf.fill(0)
            his.fill(0)
            pdf.fill(0)
            
            snap = img_temp[i-row_top: i+row_bottom+1, j-col_left: j+col_right+1].copy()
            for x in range(0,len(snap)):
                for y in range(0,len(snap[x])):

                    his[snap[x][y]] += 1

            pdf[:] = his[:] / snap.size
               
            cdf[0] = pdf[0]
            
            for a in range(1,snap[row_top][col_left]+1):
                cdf[a] = pdf[a] + cdf[a - 1]
                
            out_img[i-row_top][j-col_left] = cdf[snap[row_top][col_left]] * MAX
            
    return out_img

def unsharp_mask(img, r, k):
    
    if img.dtype != np.uint8:
        return img
    
    out_img = img.copy()
    img_temp = img.copy()
    
    gauss = gaussian_kernel(1, r)
    gauss_sum = np.sum(gauss)
    
    img_blur = bf.spatial_filter(img, gauss / gauss_sum)
    
    out_img = img_temp + k * (img_temp - img_blur)
    
    return np.array(np.round(out_img), np.uint8)

def laplacian_sharpen(img, k):
    
    out_img = img.copy()
    img_temp = img.copy()
    
    laplace = np.array([[-1, -1, -1],
                        [-1, 8, -1],
                        [-1, -1, -1]])
    
#    laplace = laplace_gaussian_kernel(1, 1)
    img_lpc = bf.spatial_filter(img, laplace)
    
    out_img = img_temp - k * img_lpc
    
    return np.array(out_img, np.uint8)
