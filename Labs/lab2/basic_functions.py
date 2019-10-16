# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 15:04:22 2019

@author: Michael
"""

import cv2
from matplotlib import pyplot as plt


def display_histogram(img):
    plt.hist(img.ravel(), 256, [0, 256])
    plt.show()


def display_image(img):
    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def display_multiple_images(img):
    for i in range(len(img)):
        cv2.imshow('image ' + str(i + 1), img[i])

    cv2.waitKey(0)
    cv2.destroyAllWindows()