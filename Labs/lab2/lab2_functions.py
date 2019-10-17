import numpy as np
import cv2

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
    if a == 1 & b == 1:
        
        I = W*F
        return I
    
#    finding if column of kernel is odd or even and establishing the padding 
#    parameters for the padded array
    
    col_right, col_left = padding_param(b)
    row_bottom, row_top = padding_param(a)
    
#    creating a padded array and an output for the convolution operation 
    F_temp = np.zeros([m+col_left+col_right,n+row_top+row_bottom])
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
                    sum += snap[l][k] * W[l][k]
            I[i-row_top][j-col_left] = sum
            
    return I

# This filter applies a Non-Maximum Suppression mask twice to the image, once in a vertical manner and
# another in a horizontal direction.
def non_max_suppress(img, H, W):
    
    m,n = img.shape
    
    if H == 1 & W == 1:
        return img
    
#    establishing the padding parameters for the padded array       
    col_right, col_left = padding_param(W)
    row_bottom, row_top = padding_param(H)
        
    # creating a padded array and an output for the max operation
    F_temp = np.zeros([m+col_left+col_right,n+row_top+row_bottom])
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
                
            else:
                I_horizontal[i-row_top,j-col_left] = 0
            
#    same operation as above except with a vertical kernel
    for i in range(row_top,m+row_top):
        for j in range(col_left,n+col_left):
            
            snap = F_temp[i, j-col_left: j+col_right+1].copy()
            if snap[col_left] == np.amax(snap):
                I_vertical[i-row_top,j-col_left] = snap[col_left]
            else:
                I_vertical[i-row_top,j-col_left] = 0
        
            
    return I_horizontal, I_vertical

# This filter applies a 2-D Non-Maximum Suppression mask throughout the 
# whole image. 
def non_max_suppress_full(img, H, W):
    
    m,n = img.shape
    
    if H == 1 & W == 1:
        return img
    
#    establishing the padding parameters for the padded array       
    col_right, col_left = padding_param(W)
    row_bottom, row_top = padding_param(H)
        
    # creating a padded array and an output for the max operation
    F_temp = np.zeros([m+col_left+col_right,n+row_top+row_bottom])
    F_temp[row_top:m+row_top, col_left:n+col_left] = 1.0
    F_temp[row_top:m+row_top, col_left:n+col_left] *= img
    I = np.zeros_like(img)
    
    for i in range(row_top,m+row_top):
        for j in range(col_left,n+col_left):
            
#            partitioning the array and computing the max value and storing
#            it in the output array if the center pixel intensity is the max
#            value
            snap = F_temp[i-row_top: i+row_bottom+1, j-col_left: j+col_right+1].copy()
            if snap[row_top, col_left] == np.amax(snap):
                I[i-row_top,j-col_left] = snap[row_top, col_left]
                
    return I
            

def image_thresholding(img, T):
    
    if 0 > T or T > 1:
        return img
    
    max_value = np.iinfo(img.dtype).max
    
#    Checks for the max value of the data type the image uses and multiplies
#    it with the ratio
    T *= max_value
    
    B = np.zeros_like(img)
    
    for i in range(0,len(img)):
        for j in range(0,len(img[i])):
            
            if img[i,j] >= T:
                B[i,j] = max_value
    return B


def edge_detector(img, H, T=0.1, wndsz=5):
    
     """Function that detect edges in an image .
    
     Input Parameters :
     
     img: image being processed ( can be either greyscale or RGB ).
     
     H: The filtering kernel that approximates the horizontal derivative.
     
     T [ optional ]: The threshold value used by the edge detector ( default 
     value is 0.1).
     
     wndsz [ optional ]: the size of the NMS filter window ( default is 5).
    
     Output:
     a binary image where a value of ’1 ’ indicates an image edge.
     """
     
#     Converts the image to a grayscale if it isn't already
     if(img.size == 3):
         I = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2GRAY)
     else:
         I = img.copy()
     
#     Gaussian Kernel to reduce noise
     g_kernel = np.array([[1, 4, 7, 4, 1],
            [4, 20, 33, 20, 4],
            [7, 33, 55, 33, 7],
            [4, 20, 33, 20, 4],
            [1, 4, 7, 4, 1]])

     g_sum = np.sum(g_kernel)   

#     Gets the transpose of the horizontal kernel to get the vertical kernel
     H_t = np.transpose(H)
     
#     used to reduce noise in the image while preserving edges
     I = spatial_filter(I, g_kernel / g_sum)  
     
#     Convolutes the derivative approximation kernels to find the image 
#     gradients
     I_x, I_y = spatial_filter(I, H), spatial_filter(I, H_t)
     
#     Computes the gradient magnitude
     I = np.sqrt(I_x**2 + I_y**2)
     
#     Suppressing low fluctuations in intensities
     I = non_max_suppress_full(I, wndsz, wndsz)
     
#     Threshold all small values that indicate weak edges
     I = image_thresholding(np.array(I, np.uint8), T)
     
     return I
     
def derivative_kernel(img, select):
    
    """Function that provides 4 derivative approximation kernels that can be 
    individually selected to be convoluted over the image. Then, the 
    convoluted image will be returned.
    
    Input Parameters :
    img (array, type = uint8): Image matrix that will be convoluted over
    
    select (int): Selects the operation to be used. Integer values between 0-3
    will select. Values not within that range will return the original image.
    
    select values
    [0] Central Difference: h = [1, 0, -1]
    
    [1] Forward Difference: h = [0, 1, -1]
    
    [2] Prewitt           : h = [[1, 0, -1],
                                 [1, 0, -1],
                                 [1, 0, -1]]
    
    [3] Prewitt           : h = [[1, 0, -1],
                                 [2, 0, -2],
                                 [1, 0, -1]]
    
    Output: An post-processed image
    """
    
#    checks for boundary conditions
    select = int(select)
    if  select > 3 or select < 0:
        return img
    
#    selects operation
    if select == 0:
        h = [1, 0, -1]
    
    elif select == 1:
        h = [0, 1, -1]
    
    elif select == 2:
        h = [[1, 0, -1],
             [1, 0, -1],
             [1, 0, -1]]
    else:
        h = [[1, 0, -1]
             [2, 0, -2]
             [1, 0, -1]]
        
#    uses the spatial_filter function to implement the operations
    return spatial_filter(img, h)
    
    
    