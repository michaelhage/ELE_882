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

#Produces binary threshold image
def simple_threshold(img, cutoff):
    
    out = np.zeros_like(img)
    
    for i in range(0, len(img)):
        for j in range(0, len(img[i])):
            
            if img[i][j] > cutoff:
                out[i][j] = 255
    
    return out

#Implements a soft thresholding function
def soft_thereshold(img, cutoff, phi):
    
    out = np.array(np.ones_like(img), np.double)
    
    for i in range(0, len(img)):
        for j in range(0, len(img[i])):
            
            if img[i][j] <= cutoff:
                out[i][j] = 1 + np.tanh(phi * (img[i][j]) - cutoff)
            
    return np.array( (out * 127) + 1, np.uint8)

def three_tone(img, cutoff, phi, k, sigma, p):
    
    out = XDOG(img, k, sigma, p)
    
    out = soft_thereshold(out, cutoff, phi)
    
    return out

def oilify(img, R, gamma):
    
    N = np.iinfo(img.dtype).max
    m,n = img.shape
    out = np.zeros_like(img)
    
    col_right, col_left = bf.padding_param(R)
    row_bottom, row_top = bf.padding_param(R)
    
    img_temp = np.array(np.zeros([m+row_top+row_bottom,n+col_left+col_right]), np.uint8)
    img_temp[row_top:m+row_top, col_left:n+col_left] = 1
    img_temp[row_top:m+row_top, col_left:n+col_left] *= img
    
    h = np.zeros(N+1)
    acc = np.zeros(N+1)
    
    for i in range(row_top,m+row_top):
        for j in range(col_left,n+col_left):
            
            snap = img_temp[i-row_top: i+row_bottom+1, j-col_left: j+col_right+1].copy()
            
            h[:] = 0
            acc[:] = 0
            
            for x in range(0,len(snap)):
                for y in range(0,len(snap[x])):
                    
                    h[snap[x][y]] += 1
                    acc += snap[x][y]
            
            h_max = np.amax(h)
            A = 0; B = 0
            
            for i in range(0, N+1):
                w = (h[i] / h_max) ** gamma
                B += w
                A += w * (acc[i] / h[i])
            
            print (A, B)
            out[i][j] = int(A / B)