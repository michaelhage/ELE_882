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
    
#    Creation of the gaussian kernels
    gauss_1 = gaussian_kernel(sigma, 2)
    gauss_2 = gaussian_kernel(k * sigma, 2)
    
    gauss_1_sum = np.sum(gauss_1)
    gauss_2_sum = np.sum(gauss_2)
    
#    Both gaussian kernels are applied to the image
    G1 = bf.spatial_filter(img_temp, gauss_1 / gauss_1_sum)
    G2 = bf.spatial_filter(img_temp, gauss_2 / gauss_2_sum)
    
#    Difference of Gaussian Computation
    img_temp[:,:] = (1 + p) * G1[:,:] - p * G2[:,:]
    
    return np.array(img_temp, np.uint8)

#Produces binary threshold image
def hard_threshold(img, cutoff):
    
    out = np.zeros_like(img)
    
#    applies the max value to the intensities greater than the cutoff
    out[img > cutoff] = 255
    
    return out

#Implements a soft thresholding function
def soft_thereshold(img, cutoff, phi):
    
    out = np.array(np.ones_like(img), np.double)
    
    for i in range(0, len(img)):
        for j in range(0, len(img[i])):
            
            if img[i][j] <= cutoff:
                out[i][j] = 1 + np.tanh(phi * (img[i][j] - cutoff))
            
    return np.array( out * 255, np.uint8)

# Three tone generator
def three_tone(img, cutoff, phi):
    
    out = np.array(np.ones_like(img), np.double)
    
    for i in range(0, len(img)):
        for j in range(0, len(img[i])):
                
            out[i][j] = 1 + np.tanh(phi * (img[i][j] - cutoff) )
            
    return np.array( (out * 127) + 1, np.uint8)


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


def edge_preserving(img, max_window_size, iteration):
    
    out_img = img.copy()
    
#    Check for RGB image, if so then convert to grayscale
    if out_img.ndim == 3:         
        out_img = cv2.cvtColor(out_img, cv2.COLOR_BGR2GRAY)
        
    for i in range(0, iteration):
        out_img = cv2.medianBlur(out_img, max_window_size - 2 * i)
    
    return out_img


def threshold(img, hi, lo):
    
#    Values for strong and weak edges
    high = 255
    low = 50
    
    out_img = np.zeros_like(img)
    
    strong_x, strong_y = np.where(img >= hi)
    weak_x, weak_y = np.where((img <= hi) & (img >= lo))

    out_img[strong_x, strong_y] = high
    out_img[weak_x, weak_y] = low
    
    return np.array(out_img, np.uint8)

def hysteresis(image):
    
    weak = 50
    
    image_row, image_col = image.shape
 
    top_to_bottom = image.copy()
 
    for row in range(1, image_row - 1):
        for col in range(1, image_col - 1):
            if top_to_bottom[row, col] == weak:
                if top_to_bottom[row, col + 1] == 255 or top_to_bottom[row, col - 1] == 255 or top_to_bottom[row - 1, col] == 255 or top_to_bottom[
                    row + 1, col] == 255 or top_to_bottom[
                    row - 1, col - 1] == 255 or top_to_bottom[row + 1, col - 1] == 255 or top_to_bottom[row - 1, col + 1] == 255 or top_to_bottom[
                    row + 1, col + 1] == 255:
                    top_to_bottom[row, col] = 255
                else:
                    top_to_bottom[row, col] = 0
 
    bottom_to_top = image.copy()
 
    for row in range(image_row - 2, 1, -1):
        for col in range(image_col - 2, 1, -1):
            if bottom_to_top[row, col] == weak:
                if bottom_to_top[row, col + 1] == 255 or bottom_to_top[row, col - 1] == 255 or bottom_to_top[row - 1, col] == 255 or bottom_to_top[
                    row + 1, col] == 255 or bottom_to_top[
                    row - 1, col - 1] == 255 or bottom_to_top[row + 1, col - 1] == 255 or bottom_to_top[row - 1, col + 1] == 255 or bottom_to_top[
                    row + 1, col + 1] == 255:
                    bottom_to_top[row, col] = 255
                else:
                    bottom_to_top[row, col] = 0
 
    right_to_left = image.copy()
 
    for row in range(1, image_row - 1):
        for col in range(image_col - 2, 0, -1):
            if right_to_left[row, col] == weak:
                if right_to_left[row, col + 1] == 255 or right_to_left[row, col - 1] == 255 or right_to_left[row - 1, col] == 255 or right_to_left[
                    row + 1, col] == 255 or right_to_left[
                    row - 1, col - 1] == 255 or right_to_left[row + 1, col - 1] == 255 or right_to_left[row - 1, col + 1] == 255 or right_to_left[
                    row + 1, col + 1] == 255:
                    right_to_left[row, col] = 255
                else:
                    right_to_left[row, col] = 0
 
    left_to_right = image.copy()
 
    for row in range(image_row - 2, 1, -1):
        for col in range(1, image_col - 2):
            if left_to_right[row, col] == weak:
                if left_to_right[row, col + 1] == 255 or left_to_right[row, col - 1] == 255 or left_to_right[row - 1, col] == 255 or left_to_right[
                    row + 1, col] == 255 or left_to_right[
                    row - 1, col - 1] == 255 or left_to_right[row + 1, col - 1] == 255 or left_to_right[row - 1, col + 1] == 255 or left_to_right[
                    row + 1, col + 1] == 255:
                    left_to_right[row, col] = 255
                else:
                    left_to_right[row, col] = 0
 
    final_image = top_to_bottom + bottom_to_top + right_to_left + left_to_right
    
    for i in range(0,len(final_image)):
        for j in range(0,len(final_image[i])):
                
            if(final_image[i,j]):
                    final_image[i,j] = 255
    
    return final_image

def extract_edges(img, sigma = 0.33):
    """ 
    This will extract edges more vividly compared to the earlier edition in lab 2
    This particular one adds hysteresis to figure out connecting edges and has auto 
    thesholding with high and lower limits to remove more detailed parts of the image 
    and retain only the important aspects
    """
    
    out_img = np.zeros_like(img)
    
#    parameter for auto thresholding
    med = np.median(img)
    
#    High and Low Auto Threshold Values
    lo_thresh = int(max(0, (1.0 - sigma) * med))
    hi_thresh = int(min(255, (1.0 + sigma) * med))
    
#    Sobel Filter
    H = np.array([[1, 0, -1],
                  [2, 0, -2],
                  [1, 0, -1]])
    
#    Transposed Sobel
    H_t = np.transpose(H)
    
#    Gaussian Kernel
    gauss = gaussian_kernel(1, 2)
    gauss_sum = np.sum(gauss)
    
#    Noise Reduction Step, Gaussian Blur
    img_temp = bf.spatial_filter(img, gauss / gauss_sum)
    
    I_x = bf.spatial_filter(img_temp, H)
    I_y = bf.spatial_filter(img_temp, H_t)
     
#    Compute gradient magnitude
    I = np.sqrt(I_x**2 + I_y**2)
     
#    Suppressing low fluctuations in intensities
    I_x, I_y = bf.non_max_suppress(I, 5, 5)
    
    I_x = threshold(I_x, hi_thresh, lo_thresh)
    I_y = threshold(I_y, hi_thresh, lo_thresh)
    
    for i in range(0,len(I_x)):
        for j in range(0, len(I_x[i])):
             
            if I_x[i,j] == 255 or I_y[i,j] == 255:
                out_img[i,j] = 255
            
            elif I_x[i,j] == 50 or I_y[i,j] == 50:
                out_img[i,j] = 50
            
            elif(I_x[i,j] == 50 or I_y[i,j] == 255) or (I_x[i,j] == 255 or I_y[i,j] == 50):
                out_img[i,j] = 50
    
    out_img = hysteresis(out_img)
    
    return np.array(out_img, np.uint8)


def cartoon_effect(img, max_window_size, iteration, sigma = 0.33, k = 1.4, p = 1, sigma_x = 1, flag = 0):
    """
    This function implements the cartoon effect filter
    
    This filter has two modes
    Select them by either setting the value of the flag to
    0 - Regular
    1 - XDOG mode
    
    Regular mode takes the standard parameters and sigma(if needed) and uses 
    the edge detector and the edge preserving smoothing filter.
    
    XDOG mode incorporates an XDOG filter in order to amplify the cartoon 
    effect.
    """
    
    img_edge = extract_edges(img, sigma = sigma)
    
    img_blur = edge_preserving(img, max_window_size, iteration)
    
    out_img = img_blur
    
#    Incorporation of the XDOG function
    if(flag == 1):
#       Creation of the gaussian kernels
        gauss_1 = gaussian_kernel(sigma_x, 2)
        gauss_2 = gaussian_kernel(k * sigma_x, 2)
    
        gauss_1_sum = np.sum(gauss_1)
        gauss_2_sum = np.sum(gauss_2)
    
#       Both gaussian kernels are applied to the image
        G1 = bf.spatial_filter(img, gauss_1 / gauss_1_sum)
        G2 = bf.spatial_filter(img, gauss_2 / gauss_2_sum)
    
    for i in range(1, len(img) - 1):
        for j in range(1, len(img[i]) - 2):
            
            if(img_edge[i, j] == 255):
                
#                Calculate the average surrounding pixel intensities except the edge pixel
                sum = np.sum(img[i-1:i+2, j-1:j+2])
                out_img[i,j] = (sum - img[i,j]) / 8

#            Implementing the XDOG function if the flag has been set
            if(flag == 1):
                out_img[i,j] += p * (G1[i,j] - G2[i,j])
                
    return out_img