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

def display_color_histogram(img):
    
    color = ('b','g','r')
    for i,col in enumerate(color):
        plt.hist(img[:,:,i].ravel(), 256, [0, 256], color = col)
        plt.show()
    
#def display_individual_histograms(img):
#    i=0
#    color = ('b','g','r')
#    
#    histr = cv2.calcHist([img],[i],None,[256],[0,256])
#    plt.plot(histr,color = color[i])
#    plt.xlim([0,256])
#    plt.show()
#    
#    i += 1
#    
#    histr = cv2.calcHist([img],[i],None,[256],[0,256])
#    plt.plot(histr,color = color[i])
#    plt.xlim([0,256])
#    plt.show()
#    
#    i += 1
#    
#    histr = cv2.calcHist([img],[i],None,[256],[0,256])
#    plt.plot(histr,color = color[i])
#    plt.xlim([0,256])
#    plt.show()
    

def display_multiple_images(img):
    for i in range(len(img)):
        cv2.imshow('image ' + str(i + 1), img[i])

    cv2.waitKey(0)
    cv2.destroyAllWindows()