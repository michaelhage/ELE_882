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
            A = 0 
            B = 0            
            
            for k in range(0, N+1):
                if(h[k] != 0):
                    w = (h[k] / h_max) ** gamma
                    B += w
                    A += w * (acc[k] / h[k])
            
            out[i-row_top][j-col_left] = A / B
            
    return np.array(out, np.uint8)


def edge_preserving(img, min_window_size, iteration):
    
    out_img = img.copy()
    
#    Check for RGB image, if so then convert to grayscale
    if out_img.ndim == 3:
        out_img = cv2.cvtColor(out_img, cv2.COLOR_BGR2GRAY)
    
    for i in range(0, iteration):
        out_img = cv2.medianBlur(out_img, min_window_size + 2 * i)
    
    return out_img

def extract_edges(img, threshold):
    
    out_img = np.zeros_like(img)
    
    gauss = gaussian_kernel(1, 2)
    gauss_sum = np.sum(gauss)
    
    laplace = np.array([[-1, -1, -1],
                        [-1, 8, -1],
                        [-1, -1, -1]])
    
    img_temp = bf.spatial_filter(img, gauss / gauss_sum)
    
    img_temp = bf.spatial_filter(img_temp, laplace)
    
    
    for i in range(1,len(img_temp)-1):
        for j in range(1,len(img_temp[i])-1):
            
            if(np.absolute(img_temp[i][j]) > threshold):
                
                if(img_temp[i-1][j] < 0 and img_temp[i+1][j] > 0) or (img_temp[i-1][j] > 0 and img_temp[i+1][j] < 0):
                    out_img[i][j] += 1
                
                if(img_temp[i][j-1] < 0 and img_temp[i][j+1] > 0) or (img_temp[i][j-1] > 0 and img_temp[i][j+1] < 0):
                    out_img[i][j] += 1
                
                if(img_temp[i+1][j+1] < 0 and img_temp[i-1][j-1] > 0) or (img_temp[i+1][j+1] > 0 and img_temp[i-1][j-1] < 0):
                    out_img[i][j] += 1
                
                if(img_temp[i-1][j+1] < 0 and img_temp[i+1][j-1] > 0) or (img_temp[i-1][j+1] > 0 and img_temp[i+1][j-1] < 0):
                    out_img[i][j] += 1
                
                if(out_img[i][j] > 2):
                    out_img[i][j] = 255
    
    return np.array(out_img, np.uint8)