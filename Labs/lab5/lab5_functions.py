# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 09:20:29 2019

@author: Michael
"""

import numpy as np
import basic_functions as bf
import cv2
import math

def padding_param(a):
    
    if a % 2 == 1:
        m = int((a - 1)/ 2)
        n = int(m)
    
    else:
        m = int(a / 2)
        n = int(m - 1)
    
    return m,n

def gaussian_kernel(sigma, r):
    
    size = 2 * r + 1
    
    kernel = np.zeros([size, size])
    
    k = (2 * sigma**2)
    k_pi = k / np.pi
    
    for i in range(0, size):
        for j in range(0, size):
            
            kernel[i][j] = k_pi * np.exp( -( (i - r)**2 + (j - r)**2 ) / k )
    
    kernel = kernel / kernel[0][0]
    
    return np.array(kernel, np.uint8), np.sum(kernel)

def rgb_to_hsi(img):
    
    img_temp = np.array(img.copy(), np.double)
    out_img = np.array(np.zeros_like(img), np.double)
    
    r_arr = img_temp[:,:,2] / 255
    g_arr = img_temp[:,:,1] / 255
    b_arr = img_temp[:,:,0] / 255
    
    for i in range(0,len(img)):
        for j in range(0,len(img[i])):
            
            r = r_arr[i,j]
            g = g_arr[i,j]
            b = b_arr[i,j]
            
#            Hue
            a = (0.5 * ( (r - g) + (r - b) ) ) / np.sqrt( (r - g)**2 + (r - b ) * ( g - b ) )
            
            if np.isnan(a):
                a = 0
            
            out_img[i,j,0] = math.acos( a )
            
            if(b > g):
                out_img[i,j,0] = 2 * np.pi - out_img[i,j,0]
            
#            Saturation
            c_min = np.min([r, g, b])
            
            if r+g+b == 0:
                out_img[i,j,1] = 1
            else:
                out_img[i,j,1] = 1 - ( (3.0 / (r + g + b) ) * c_min)
            
#            Intensity
            out_img[i,j,2] = (r + g + b) / 3
    
    return np.array(out_img, np.double)

def hsi_to_rgb(img):
    
    out_img = np.zeros_like(img)
    
    H = img[:,:,0]
    S = img[:,:,1]
    I = img[:,:,2]
    
    for i in range(0,len(img)):
        for j in range(0,len(img[i])):
            
            
            if 0 <= H[i,j] < (2 * np.pi / 3):
                
                out_img[i,j,0] = I[i,j] * (1 - S[i,j])
                
                out_img[i,j,2] = I[i,j] * (1 + (S[i,j] * math.cos(H[i,j]) / math.cos((np.pi / 3) - H[i,j])) )
                
                out_img[i,j,1] = 3 * I[i,j] - (out_img[i,j,0] + out_img[i,j,2])
                
            elif (2 * np.pi / 3) <= H[i,j] < (4 * np.pi / 3):
                
                out_img[i,j,2] = I[i,j] * (1 - S[i,j])
                
                out_img[i,j,1] = I[i,j] * (1 + (S[i,j] * math.cos(H[i,j] - (2 * np.pi / 3)) ) / math.cos(np.pi - H[i,j]))
                
                out_img[i,j,0] = 3 * I[i,j] - (out_img[i,j,2] + out_img[i,j,1])
                
            elif (4 * np.pi / 3) <= H[i,j] < 2 * np.pi:
                
                out_img[i,j,1] = I[i,j] * (1 - S[i,j])
                
                out_img[i,j,0] = I[i,j] * (1 + (S[i,j] * math.cos(H[i,j] - (4 * np.pi / 3))) / math.cos((5 * np.pi / 3) - H[i,j]))
                
                out_img[i,j,2] = 3 * I[i,j] - (out_img[i,j,1] + out_img[i,j,0])
        
    return np.array(out_img * 255, np.uint8)


def rgb_to_ycbcr(img):
    
    m = np.array([ [0.299, -0.1687, 0.5],
                   [0.587, -0.3313, -0.4187],
                   [0.114, 0.5, -0.0813]])
    
    out_img = np.dot(img, m)
    out_img[:,:,1:] += 128.0
    
    return out_img


def ycbcr_to_rgb(img):
    
    m = np.array([[1.0, 1.0, 1.0],
                  [0, -0.34414, 1.772],
                  [1.402, -0.71414, 0]])
    
    
    out_img = np.array(img.copy(), np.double)
    
    out_img = img.dot(m) 
    
#    These offsets are required because the values are stored into 8-bits
    
#    Blue Channel -128 * 1.402 = -180.096
    out_img[:,:,0] -= 180.096
    
#    Green Channel (-128 * -0.34414) + (-128 * -0.71414) = 135.459
    out_img[:,:,1] += 135.459
    
#    Red Channel -128 * 1.772 = -226.816
    out_img[:,:,2] -= 226.816
    
    return np.array(out_img, np.uint8)

def change_hue(img, hue_angle):
    """
    This is a function that will rotate the hue of all values in an HSI image map.
    
    image: input color image that uses the BGR mapping
    
    hue_angle: input angle, in radians, that rotates the hue
    
    return: output image of rotated hue that uses BGR mapping
    """
    
    img_temp = img.copy()
    
    img_temp = rgb_to_hsi(img_temp)
    
    tau = 2 * np.pi
    
    img_temp[:,:,0] = (img_temp[:,:,0] + hue_angle) % tau
    
    return hsi_to_rgb(img_temp)

def change_saturation(img, sat):
    
    img_temp = img.copy()
    
    img_temp = rgb_to_hsi(img_temp)
    
    img_temp[:,:,1] += sat
    
    img_temp[img_temp[:,:,1] > 1] = 1
    img_temp[img_temp[:,:,1] < 0] = 0

    return hsi_to_rgb(img_temp)

def apply_point_tfrm(img, c, d):
    
    out_img = np.zeros_like(img)
    
    # this copies the image independent of the original image
    r = img[:,:,2]
    g = img[:,:,1]
    b = img[:,:,0]

    # iterates through the array
    for i in range(len(img)):
        for j in range(len(img[i])):

            # applies the transformation
            x = (c * b[i,j]) + d
            y = (c * g[i,j]) + d
            z = (c * r[i,j]) + d


            # checks for bit overflow
            if x > 255:
                x = 255
            elif x < 0:
                x = 0
            
            if y > 255:
                y = 255
            elif y < 0:
                y = 0
            
            if z > 255:
                z = 255
            elif z < 0:
                z = 0

            # applies the pixel value to the image
            out_img[i,j,0] = x
            out_img[i,j,1] = y
            out_img[i,j,2] = z

    return out_img


def spatial_filter(img, W): 
    
    if img.dtype != np.uint8:
        return img
    
    a,b = W.shape
    m,n,x = img.shape
    
    img_double = np.array(img.copy(), np.double)
    out_img = np.zeros_like(img_double)
    
#    if filter kernel has size 1 by 1
    if a == 1 and b == 1:
        
        return W*img_double
    
#    finding if column of kernel is odd or even and establishing the padding 
#    parameters for the padded array
    
    col_right, col_left = padding_param(a)
    row_bottom, row_top = padding_param(b)
    for x in range(0,x):
#        creating a padded array and an output for the convolution operation 
        img_temp = np.zeros([m+row_top+row_bottom,n+col_left+col_right])
        img_temp[row_top:m+row_top, col_left:n+col_left] = 1.0
        
        img_temp[row_top:m+row_top, col_left:n+col_left] *= img_double[:,:,x]
        
#        iterating over the length and width of the original size of the array
        for i in range(row_top,m+row_top):
            for j in range(col_left,n+col_left):
                
                sum = 0
#                partioning a section the same size as the kernel for the 
#                convoltion operation and then computing the convolution and
#                storing it in the output array
              
                snap = img_temp[i-row_top: i+row_bottom+1, j-col_left: j+col_right+1].copy()
                for l in range(0,len(W)):
                    for k in range(0,len(W[l])):
                        
                        sum += snap[l,k] * W[l,k]
                        
                out_img[i-row_top,j-col_left,x] = sum
                
    return np.array(out_img, np.uint8)


def histogram_equalization_rgb(img):
    
#    Check for unsigned integer 8-bit
    if img.dtype != np.uint8:
        return img
    
    out_img = img.copy()
    MAX = np.iinfo(np.uint8).max
    m,n,c = img.shape
    
#    Creates the histogram
    his = np.zeros(MAX+1)
    cdf = np.zeros(MAX+1)
    
    for x in range(0,c):
        his[:] = 0
        cdf[:] = 0
        
#        Calculate the histogram
        for i in range(0, len(img)):
            for j in range(0, len(img[i])):
                
                his[ img[i,j,x] ] += 1
         
#        Calculate the CDF    
        cdf[0] = his[0]
        
        for i in range(1,len(cdf)):
            cdf[i] = his[i] + cdf[i - 1]
        
        cdf = cdf / (m*n) * MAX
        
#        Apply the CDF
        for i in range(0, len(img)):
            for j in range(0, len(img[i])):
                
                out_img[i,j,x] = cdf[img[i,j,x]]
    
    return out_img


def histogram_equalization_ycbcr(img):
    
#    Check for unsigned integer 8-bit
    if img.dtype != np.uint8:
        return img
    
    img_temp = rgb_to_ycbcr(img)
    img_temp = np.array(img_temp, np.uint8)
    
    
    out_img = img_temp.copy()
    MAX = 255
    m,n,c = img.shape
    
#    Creates the histogram
    his = np.zeros(MAX+1)
    cdf = np.zeros(MAX+1)

#    Calculate the histogram
    for i in range(0, len(img)):
        for j in range(0, len(img[i])):
            
            his[ img_temp[i,j,0] ] += 1
     
#    Calculate the CDF    
    cdf[0] = his[0]
    
    for i in range(1,MAX+1):
        cdf[i] = his[i] + cdf[i - 1]
    
    cdf = cdf / (m*n) * MAX
    
#    Apply the CDF
    for i in range(0, len(img)):
        for j in range(0, len(img[i])):
            
            out_img[i,j,0] = cdf[img_temp[i,j,0]]
    
    
    
    out_img = ycbcr_to_rgb(np.uint8(out_img))
    
    return out_img