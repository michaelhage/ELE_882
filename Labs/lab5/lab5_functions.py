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
    
    r = img[:,:,0] / 255
    g = img[:,:,2] / 255
    b = img[:,:,1] / 255
    
#    Hue
    out_img[:,:,0] = np.arccos( (0.5 * (2*r[:,:] - g[:,:] - b[:,:])) / np.sqrt( (r[:,:]-g[:,:])**2 + (r[:,:]-b[:,:])*(g[:,:]-b[:,:])) )
    
#    Saturation
    c_min = np.min(img, axis = 2) / 255
    
    out_img[:,:,1] = 1 - (3 / (r[:,:] + g[:,:] + b[:,:])) * c_min
    
#    Intensity
    out_img[:,:,2] = (r[:,:] + g[:,:] + b[:,:]) / 3
    
    return np.array(out_img, np.double)

def hsi_to_rgb(img):
    
    H = img[:,:,0]
    S = img[:,:,1]
    I = img[:,:,2]
    
    out_img = np.zeros_like(img)
    
    for i in range(0,len(img)):
        for j in range(0,len(img[i])):
            
            if 0 <= H[i,j] < (2 * np.pi / 3):
                
                out_img[i,j,1] = H[i,j] * (1 - S[i,j])
                
                out_img[i,j,0] = I[i,j] * (1 + (S[i,j] * np.cos(H[i,j]) / np.cos((np.pi / 3) - H[i,j])) )
                
                out_img[i,j,2] = 3 * I[i,j] - (out_img[i,j,0] + out_img[i,j,2])
                
            elif (2 * np.pi / 3) <= H[i,j] < (4 * np.pi / 3):
                
                out_img[i,j,0] = I[i,j] * (1 - S[i,j])
                
                out_img[i,j,2] = I[i,j] * (1 + (S[i,j] * np.cos(H[i,j] - (2 * np.pi / 3)) ) / np.cos(np.pi - H[i,j]))
                
                out_img[i,j,1] = 3 * I[i,j] - (out_img[i,j,0] + out_img[i,j,2])
                
            elif (4 * np.pi / 3) <= H[i,j] < 2 * np.pi:
                
                out_img[i,j,2] = I[i,j] * (1 - S[i,j])
                
                out_img[i,j,1] = I[i,j] * (1 + (S[i,j] * np.cos(H[i,j] - (4 * np.pi / 3))) / np.cos((5 * np.pi / 3) - H[i,j]))
                
                out_img[i,j,0] = 3 * I[i,j] - (out_img[i,j,1] + out_img[i,j,2])
        
    return np.array(out_img * 255, np.uint8)