import cv2
import numpy as np
import lab1_functions as fl

# Section 2.1

# # Question 1
#
# # Image paths
# photo_path_1 = "Images/Section2.1 - Q1/183087.jpg"
# photo_path_2 = "Images/Section2.1 - Q1/207056.jpg"
#
# # Import image
# img_1 = cv2.imread(photo_path_1, 0)
#
# # define parameters C and B
# c = 1
# b = -80
#
# # Point Transformation
# out_img = fl.apply_point_tfrm(img_1, c, b)
#
# # Display both images
# fl.display_multiple_images([img_1, out_img])
#
#
# # Question 2
#
# # Image paths
# photo_path_1 = "Images/Section2.1 - Q2/bridge.png"
# photo_path_2 = "Images/Section2.1 - Q2/fish.png"
# photo_path_3 = "Images/Section2.1 - Q2/mask.png"
#
# # adjust path 1, 2 and 3
# img_1 = cv2.imread(photo_path_1, 0)
# img_2 = cv2.imread(photo_path_2, 0)
# img_mask = cv2.imread(photo_path_3, 0)
#
# # apply mask
# out_img = fl.apply_mask(img_1, img_2, img_mask)
#
# # display the images
# img_arr = [img_1, img_2, img_mask, out_img]
# fl.display_multiple_images(img_arr)
#
#
# # Question 3
#
# # Image paths
# photo_path_1 = "Images/Section2.1 - Q3/snowglobe_001.png"
# photo_path_2 = "Images/Section2.1 - Q3/snowglobe_002.png"
# photo_path_3 = "Images/Section2.1 - Q3/snowglobe_003.png"
# photo_path_4 = "Images/Section2.1 - Q3/snowglobe_004.png"
#
#
# img_1 = cv2.imread(photo_path_1, 0)
# img_2 = cv2.imread(photo_path_2, 0)
# img_3 = cv2.imread(photo_path_3, 0)
# img_4 = cv2.imread(photo_path_4, 0)
#
# # array of all the images
# img_arr = [img_1, img_2, img_3, img_4]
#
# # Appends the output to the image array
# # takes the output array from the average_img method and converts it to an array with a data type of
# # unsigned 8-bits (np.uint8) representing the values.
# img_arr.append(np.array(fl.average_img(img_arr), np.uint8))
#
# # Display images (only the first one and the output)
# fl.display_multiple_images([img_arr[0], img_arr[4]])
#
#
# # Section 2.2
#
# # Question 1
#
# # Image paths
# photo_path_1 = "Images/Section2.2 - Q1/183087.jpg"
# photo_path_2 = "Images/Section2.2 - Q1/motion01.512.tiff"
#
# img = cv2.imread(photo_path_2, 0)
#
# out_img = fl.contrast_stretching(img)
#
# fl.display_multiple_images([img,out_img])
#
# fl.display_histogram(img)
# fl.display_histogram(out_img)
#
#
# # Question 2
#
# # Image paths
# photo_path_1 = "Images/Section2.2 - Q2/7.1.01.tiff"
# photo_path_2 = "Images/Section2.2 - Q2/7.1.02.tiff"
#
# # Define parameters
# r1 = 70
# s1 = 0
# r2 = 140
# s2 = 255
#
# img = cv2.imread(photo_path_1, 0)
#
# out_img = fl.contrast_piecewise(img, r1, s1, r2, s2)
#
# fl.display_multiple_images([img, out_img])
#
#
# # Question 3
#
# # Image paths
# photo_path_1 = "Images/Section2.2 - Q2/7.1.01.tiff"
# photo_path_2 = "Images/Section2.2 - Q2/7.2.01.tiff"
#
# # define parameters
# a = 70
# b = 150
#
# img = cv2.imread(photo_path_1, 0)
#
# out_img = fl.contrast_highlight(img, a, b)
#
# fl.display_multiple_images([img, out_img])
#
# # Question 4
#
# L = 256
# img = cv2.imread(photo_path_1, 0)
#
# #Transfer Function
# t = np.arange(0, L-1)
#
# c = 20
#
# for i in range(len(t)):
#     t[i] = c * np.log(t[i]+1)
#
# out_img = fl.contrast_tfrm_curve(img, t)
#
# fl.display_multiple_images([img, out_img])
#
# fl.display_histogram(img)
# fl.display_histogram(out_img)
#
# # Question 5
#
# # Image paths
# photo_path_1 = "Images/Section2.2 - Q1/183087.jpg"
# photo_path_2 = "Images/Section2.2 - Q1/motion01.512.tiff"
#
# # defining parameters
#
# img = cv2.imread(photo_path_2, 0)
#
# out_img = fl.contrast_streching_LUT(img)
#
# fl.display_multiple_images([img, out_img])

# # Question 6
#
# # Contrast Piecewise
#
# # Image paths
# photo_path_1 = "Images/Section2.2 - Q2/7.1.01.tiff"
# photo_path_2 = "Images/Section2.2 - Q2/7.1.02.tiff"
#
# # Define parameters
# r1 = 70
# s1 = 0
# r2 = 140
# s2 = 255
#
# img = cv2.imread(photo_path_1, 0)
#
# out_img = fl.contrast_piecewise_LUT(img, r1, s1, r2, s2)
#
# fl.display_multiple_images([img, out_img])


# # Contrast highlight
#
# # Image path
# photo_path_1 = "Images/Section2.2 - Q3/3096.jpg"
# photo_path_2 = "Images/Section2.2 - Q3/208001.jpg"
#
# # Defining parameters
# img = cv2.imread(photo_path_2, 0)
# a = 130
# b = 255
#
# out_img = fl.contrast_highlight_LUT(img, a, b)
#
# fl.display_multiple_images([img, out_img])
