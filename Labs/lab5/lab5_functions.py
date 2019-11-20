# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 09:20:29 2019

@author: Michael
"""

import numpy as np
import basic_functions as bf
import cv2

def rgb_to_hsi(img):
    
    out_img = np.array(img.copy(), np.double)
    
    r_arr = img[:,:,2] / 255
    g_arr = img[:,:,1] / 255
    b_arr = img[:,:,0] / 255
    
    for i in range(0,len(img)):
        for j in range(0,len(img[i])):
            
            r = r_arr[i,j]
            g = g_arr[i,j]
            b = b_arr[i,j]
            
#            Hue
            a = (0.5 * ( (r - g) + (r - b) ) ) / np.sqrt( (r - g)**2 + (r - b ) * ( g - b ) )
            
            out_img[i,j,0] = np.arccos( a )
            
            if(b > g):
                out_img[i,j,0] = 2 * np.pi - out_img[i,j,0]
            
#            Saturation
            c_min = np.min(img[i,j]) / 255
            
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
                
                out_img[i,j,2] = I[i,j] * (1 + (S[i,j] * np.cos(H[i,j]) / np.cos((np.pi / 3) - H[i,j])) )
                
                out_img[i,j,1] = 3 * I[i,j] - (out_img[i,j,0] + out_img[i,j,2])
                
            elif (2 * np.pi / 3) <= H[i,j] < (4 * np.pi / 3):
                
                out_img[i,j,2] = I[i,j] * (1 - S[i,j])
                
                out_img[i,j,1] = I[i,j] * (1 + (S[i,j] * np.cos(H[i,j] - (2 * np.pi / 3)) ) / np.cos(np.pi - H[i,j]))
                
                out_img[i,j,0] = 3 * I[i,j] - (out_img[i,j,2] + out_img[i,j,1])
                
            elif (4 * np.pi / 3) <= H[i,j] < 2 * np.pi:
                
                out_img[i,j,1] = I[i,j] * (1 - S[i,j])
                
                out_img[i,j,0] = I[i,j] * (1 + (S[i,j] * np.cos(H[i,j] - (4 * np.pi / 3))) / np.cos((5 * np.pi / 3) - H[i,j]))
                
                out_img[i,j,2] = 3 * I[i,j] - (out_img[i,j,1] + out_img[i,j,0])
        
    return np.array(out_img * 255, np.uint8)


def rgb_to_ycbcr(img):
    
    out_img = np.array(img.copy(), np.double)
    
    r_arr = img[:,:,2] / 255
    g_arr = img[:,:,1] / 255
    b_arr = img[:,:,0] / 255
    
    out_img = np.zeros_like(img)
    
    for i in range(0,len(img)):
        for j in range(0,len(img[i])):
            
            r = r_arr[i,j]
            g = g_arr[i,j]
            b = b_arr[i,j]
            
            if(img.dtype == np.uint8):
                
                out_img[i,j,0] = 0.299 * r + 0.587 * g + 0.114 * b
                
                out_img[i,j,1] = -0.1687 * r - 0.3313 * g + 0.5 * b + 128
                
                out_img[i,j,2] = 0.5 * r - 0.4187 * g - 0.0813 * b + 128
            
            
            elif(img.dtype == np.double):
                
                out_img[i,j,0] = 0.299 * r + 0.587 * g + 0.114 * b
                
                out_img[i,j,1] = -0.1687 * r - 0.3313 * g + 0.5 * b
                
                out_img[i,j,2] = 0.5 * r - 0.4187 * g - 0.0813 * b
    
    return np.array(out_img, np.double)


def ycbcr_to_rgb(img):
   
    out_img = np.zeros_like(img)
    
    Y_arr = img[:,:,0]
    Cb_arr = img[:,:,1]
    Cr_arr = img[:,:,2]
    
    for i in range(0,len(img)):
        for j in range(0,len(img[i])):
            
            y = Y_arr[i,j]
            cb = Cb_arr[i,j]
            cr = Cr_arr[i,j]
            
            if(img.dtype == np.uint8):
                
                out_img[i,j,0] = 1 * y +  0.114 * (cr - 128)
                
                out_img[i,j,1] = 1 * y - 0.34414 * (cb - 128) - 0.71414 * (cr - 128)
                
                out_img[i,j,2] = 1 * y - 1.772 * (cb - 128)
            
            
            elif(img.dtype == np.double):
                
                out_img[i,j,0] = 1 * y +  0.114 * cr
                
                out_img[i,j,1] = 1 * y - 0.34414 * cb - 0.71414 * cr
                
                out_img[i,j,2] = 1 * y - 1.772 * cb
    
    return np.array(out_img, np.uint8)