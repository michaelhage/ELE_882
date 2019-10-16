import cv2
import numpy as np
import copy
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


def apply_point_tfrm(in_img, c, b):
    # this copies the image independent of the original image
    out_img = copy.copy(in_img)

    # iterates through the array
    for i in range(len(in_img)):
        for j in range(len(in_img[i])):

            # applies the transformation
            x = (c * in_img[i][j]) + b

            # checks for bit overflow
            if x > 255:
                x = 255
            elif x < 0:
                x = 0

            # applies the pixel value to the image
            out_img[i][j] = x

    return out_img


def apply_mask(img_a, img_b, img_mask):
    out_img = copy.copy(img_a)

    for i in range(len(img_a)):
        for j in range(len(img_a[i])):

            if img_mask[i][j] == 0:
                out_img[i][j] = img_b[i][j]

    return out_img


def average_img(img_arr):
    # defining length of the output image
    out_img = np.zeros((len(img_arr[0]), len(img_arr[0][0])))

    # iterate through the array
    for i in range(len(img_arr)):

        # copies the image to a temp value
        temp_img = copy.copy(img_arr[i])
        for x in range(len(temp_img)):

            for y in range(len(temp_img[x])):
                # adds the pixel values to the output array
                out_img[x][y] += temp_img[x][y]

    # takes the average value of the array
    out_img = out_img / len(img_arr)

    return out_img


def contrast_stretching(img):
    out_img = copy.copy(img)

    # get min and max intensity values
    r_min = np.min(img)
    r_max = np.max(img)

    for i in range(len(img)):

        for j in range(len(img[i])):
            # apply the transform to each pixel
            out_img[i][j] = 255 * (img[i][j] - r_min) / (r_max - r_min)

    return out_img


def contrast_piecewise(img, r1, s1, r2, s2):
    out_img = copy.copy(img)

    for i in range(len(img)):
        for j in range(len(img[i])):

            # case statements for the piecewise function
            if 0 <= img[i][j] <= r1:
                out_img[i][j] = int((s1 / r1) * img[i][j])

            elif r1 < img[i][j] <= r2:
                out_img[i][j] = int(((s2 - s1) / (r2 - r1)) * (img[i][j] - r1) + s1)

            else:
                out_img[i][j] = int(((255 - s2) / (255 - r2)) * (img[i][j] - r2) + s2)

    return out_img


def contrast_highlight(img, a, b):
    out_img = copy.copy(img)

    # defining parameters
    l = 256
    i_min = 254

    if (a + b) / 2 >= (l / 2):
        i_min = 1

    for i in range(len(img)):
        for j in range(len(img[i])):

            # checks for outside the requirement to replace with the extreme value
            if a > img[i][j] or img[i][j] > b:
                out_img[i][j] = i_min

    return out_img


def contrast_tfrm_curve(img, t):
    out_img = copy.copy(img)

    for i in range(len(img)):
        for j in range(len(img[i])):
            out_img[i][j] = t[img[i][j]]

    return out_img


def contrast_streching_LUT(img):
    # defining parameters
    l = 256
    t = np.arange(0, l)

    # finding min and max values
    r_min = np.min(img)
    r_max = np.max(img)

    # creating the LUT for the function
    for i in range(len(t)):
        t[i] = 255 * (t[i] - r_min) / (r_max - r_min)

    return contrast_tfrm_curve(img, t)


def contrast_highlight_LUT(img, a, b):
    # defining parameters
    l = 256
    i_min = 254

    if (a + b) / 2 >= (l / 2):
        i_min = 1

    # create array for LUT
    t = np.arange(0, l)

    # apply transformation to LUT
    for i in range(len(t)):
        if a > t[i] or t[i] > b:
            t[i] = i_min

    return contrast_tfrm_curve(img, t)


def contrast_piecewise_LUT(img, r1, s1, r2, s2):
    l = 256
    t = np.arange(0, l)

    for i in range(len(t)):
        if 0 <= t[i] <= r1:
            t[i] = int((s1 / r1) * t[i])

        elif r1 < t[i] <= r2:
            t[i] = int(((s2 - s1) / (r2 - r1)) * (t[i] - r1) + s1)

        else:
            t[i] = int(((255 - s2) / (255 - r2)) * (t[i] - r2) + s2)

    return contrast_tfrm_curve(img, t)
