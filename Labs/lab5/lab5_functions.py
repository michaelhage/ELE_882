# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 09:20:29 2019

@author: Michael
"""

import numpy as np
import basic_functions as bf
import cv2

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
            
            out_img[i,j,0] = np.arccos( a )
            
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
    
#    out_img = np.array(img.copy(), np.double)
#    
#    r_arr = img[:,:,2]
#    g_arr = img[:,:,1]
#    b_arr = img[:,:,0]
#    
#    out_img = np.array(np.zeros_like(img), np.double)
#    
#    for i in range(0,len(img)):
#        for j in range(0,len(img[i])):
#            
#            if(img.dtype == np.uint8):
#            
#                r = r_arr[i,j]
#                g = g_arr[i,j]
#                b = b_arr[i,j]
#                
#                out_img[i,j,0] = (0.299 * r) + (0.587 * g) + (0.114 * b)
#                
#                out_img[i,j,1] = (-0.1687 * r) - (0.3313 * g) + (0.5 * b) + 128
#                
#                out_img[i,j,2] = (0.5 * r) - (0.4187 * g) - (0.0813 * b) + 128
#            
#            
#            elif(img.dtype == np.double):
#                
#                r = r_arr[i,j] 
#                g = g_arr[i,j] 
#                b = b_arr[i,j]
#                
#                out_img[i,j,0] = (0.299 * r) + (0.587 * g) + (0.114 * b)
#                
#                out_img[i,j,1] = (-0.1687 * r) - (0.3313 * g) + (0.5 * b)
#                
#                out_img[i,j,2] = (0.5 * r) - (0.4187 * g) - (0.0813 * b)
    
    m = np.array([ [0.299, -0.1687, 0.5],
                   [0.587, -0.3313, -0.4187],
                   [0.114, 0.5, -0.0813]])
    
    out_img = np.dot(img, m)
    out_img[:,:,1:] += 128.0
    
    return out_img


def ycbcr_to_rgb(img):
   
#    out_img = np.array(np.zeros_like(img), np.double)
#    
#    Y_arr = img[:,:,0]
#    Cb_arr = img[:,:,1]
#    Cr_arr = img[:,:,2]
#    
#    for i in range(0,len(img)):
#        for j in range(0,len(img[i])):
#            
#            y = Y_arr[i,j]
#            cb = Cb_arr[i,j]
#            cr = Cr_arr[i,j]
#            
#            if(img.dtype == np.uint8):
#                
#                out_img[i,j,2] = y +  1.402 * (cr - 128)
#                
#                out_img[i,j,1] = y - 0.34414 * (cb - 128) - 0.71414 * (cr - 128)
#                
#                out_img[i,j,0] = y + 1.772 * (cb - 128)
#            
#            
#            elif(img.dtype == np.double):
#                
#                out_img[i,j,2] = y +  1.402 * cr
#                
#                out_img[i,j,1] = y - 0.34414 * cb - 0.71414 * cr
#                
#                out_img[i,j,0] = y + 1.772 * cb
    
    
    m = np.array([[1.0, 1.0, 1.0],
                  [0, -0.34414, 1.772],
                  [1.402, -0.71414, 0]])
    
    
    out_img = np.dot(img, m)
    
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
    
    image: input color image that uses the HSI mapping
    
    hue_angle: input angle, in radians, that rotates the hue
    
    return: output image of rotated hue that uses BGR mapping
    """
    
    img_temp = img.copy()
    
    img_temp[:,:,0] = (img[:,:,0] + hue_angle)
    
    for i in range(0, len(img)):
        for j in range(0, len(img[i])):
            
            if(img_temp[i,j,0] >= 2 * np.pi):
                img_temp[i,j,0] = img_temp[i,j,0] - 2 * np.pi
    
    return hsi_to_rgb(img_temp)