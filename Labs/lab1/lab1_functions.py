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

    # iterates through the image
    for i in range(len(in_img)):
        for j in range(len(in_img[i])):

            # applies the transform to a value
            x = (c * in_img[i][j]) + b

            # checks for overflow
            if x > 255:
                x = 255
            elif x < 0:
                x = 0

            # applies the value to the image pixel
            out_img[i][j] = x

    return out_img


def apply_mask(img_a, img_b, img_mask):

    # copies the first image
    out_img = copy.copy(img_a)

    # iterates through the image
    for i in range(len(img_a)):
        for j in range(len(img_a[i])):

            # if mask image pixel is black, then replace pixel with the second image
            if img_mask[i][j] == 0:
                out_img[i][j] = img_b[i][j]

    return out_img


def average_img(img_arr):
    n = 0

    out_img = np.zeros((len(img_arr[0]), len(img_arr[0][0])))

    for i in range(len(img_arr)):

        n += 1
        temp_img = copy.copy(img_arr[i])
        for x in range(len(temp_img)):

            for y in range(len(temp_img[x])):
                out_img[x][y] += temp_img[x][y]

    out_img = out_img / n

    # print(out_img)

    return out_img


def contrast_stretching(img):
    out_img = copy.copy(img)

    r_min = np.min(img)
    r_max = np.max(img)

    for i in range(len(img)):

        for j in range(len(img[i])):
            out_img[i][j] = 255 * (img[i][j] - r_min) / (r_max - r_min)

    return out_img


def contrast_piecewise(img, r1, s1, r2, s2):
    out_img = copy.copy(img)

    for i in range(len(img)):
        for j in range(len(img[i])):

            if 0 <= img[i][j] <= r1:
                out_img[i][j] = int( (s1 / r1) * img[i][j] )

            elif r1 < img[i][j] <= r2:
                out_img[i][j] = int(((s2 - s1) / (r2 - r1)) * (img[i][j] - r1) + s1)

            else:
                out_img[i][j] = int(((255 - s2) / (255 - r2)) * (img[i][j] - r2) + s2)

    return out_img


def contrast_highlight(img, a, b):
    out_img = copy.copy(img)

    if (a+b)/2 > 128:
        i_min = 1
    else:
        i_min = 254

    for i in range(len(img)):
        for j in range(len(img[i])):

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

    l = 256
    t = np.arange(0, l)

    r_min = np.min(img)
    r_max = np.max(img)

    for i in range(len(t)):
        t[i] = 255 * (t[i] - r_min) / (r_max - r_min)

        if t[i] < 0:
            t[i] = 0

    return contrast_tfrm_curve(img, t)


def contrast_highlight_LUT(img, a, b):

    l = 256

    if (a+b)/2 > 128:
        i_min = 1
    else:
        i_min = 254

    t = np.arange(0, l)

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