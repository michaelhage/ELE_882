# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 15:04:22 2019

@author: Michael

Includes functions from previous labs and also includes basic image output
scripts
"""

import cv2
from matplotlib import pyplot as plt
import numpy as np

def display_histogram(img):
    plt.hist(img.ravel(), 256, [0, 256])
    plt.show()


def display(img):
    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def display_multiple_images(img):
    for i in range(len(img)):
        cv2.imshow('image ' + str(i + 1), img[i])

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
def padding_param(a):
    
    if a % 2 == 1:
        m = int((a - 1)/ 2)
        n = int(m)
    
    else:
        m = int(a / 2)
        n = int(m - 1)
    
    return m,n


def spatial_filter(F, W): 
    
    a,b = W.shape
    m,n = F.shape
    
    F_double = np.array(F.copy(), np.double)
    
#    if filter kernel has size 1 by 1
    if a == 1 and b == 1:
        
        I = W*F
        return I
    
#    finding if column of kernel is odd or even and establishing the padding 
#    parameters for the padded array
    
    col_right, col_left = padding_param(a)
    row_bottom, row_top = padding_param(b)
    
#    creating a padded array and an output for the convolution operation 
    F_temp = np.zeros([m+row_top+row_bottom,n+col_left+col_right])
    F_temp[row_top:m+row_top, col_left:n+col_left] = 1.0
    
    F_temp[row_top:m+row_top, col_left:n+col_left] *= F_double
    I = np.zeros_like(F_double)
    
#    iterating over the length and width of the original size of the array
    for i in range(row_top,m+row_top):
        for j in range(col_left,n+col_left):
            
            sum = 0
#            partioning a section the same size as the kernel for the 
#            convoltion operation and then computing the convolution and
#            storing it in the output array
            
            snap = F_temp[i-row_top: i+row_bottom+1, j-col_left: j+col_right+1].copy()
            for l in range(0,len(W)):
                for k in range(0,len(W[l])):
                    
                    sum += snap[l][k] * W[k][l]
                    
            I[i-row_top][j-col_left] = sum
            
    return I


# This filter applies a Non-Maximum Suppression mask twice to the image, once in a vertical manner and
# another in a horizontal direction.
def non_max_suppress(img, H, W):
    
    m,n = img.shape
    
    if H == 1 and W == 1:
        return img
    
#    establishing the padding parameters for the padded array       
    col_right, col_left = padding_param(W)
    row_bottom, row_top = padding_param(H)
        
    # creating a padded array and an output for the max operation
    F_temp = np.zeros([m+row_top+row_bottom,n+col_left+col_right])
    F_temp[row_top:m+row_top, col_left:n+col_left] = 1.0
    F_temp[row_top:m+row_top, col_left:n+col_left] *= img
    I_vertical = np.zeros_like(img)
    I_horizontal = np.zeros_like(img)
    
#    iterating over the original array while applying a horizontal kernel
#    with a max value filter
    for i in range(row_top,m+row_top):
        for j in range(col_left,n+col_left):
            
#            partitioning the array and computing the max value and storing
#            it in the output array
            snap = F_temp[i-row_top: i+row_bottom+1, j].copy()
            if snap[row_top] == np.amax(snap):
                I_horizontal[i-row_top,j-col_left] = snap[row_top]
            
#    same operation as above except with a vertical kernel
    for i in range(row_top,m+row_top):
        for j in range(col_left,n+col_left):
            
            snap = F_temp[i, j-col_left: j+col_right+1].copy()
            if snap[col_left] == np.amax(snap):
                I_vertical[i-row_top,j-col_left] = snap[col_left]
            
    return I_horizontal, I_vertical