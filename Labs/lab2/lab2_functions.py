import numpy as np

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
    
#    if filter kernel has size 1 by 1
    if a == 1 & b == 1:
        
        I = W*F
        return I
    
#    finding if column of kernel is odd or even and establishing the padding 
#    parameters for the padded array
    
    col_right, col_left = padding_param(b)
    row_bottom, row_top = padding_param(a)
#    if b % 2 == 1:
#        col_right = int((b-1)/2)
#        col_left = int(col_right)
#    
#    else:
#        col_left = int(b/2)
#        col_left = int(col_right - 1)
    
#    same process as above, but for the rows instead
#    if a % 2 == 1:
#        row_bottom = int((a - 1)/ 2)
#        row_top = int(row_bottom)
#    
#    else:
#        row_bottom = int(a / 2)
#        row_top = int(row_bottom - 1)
    
#    creating a padded array and an output for the convolution operation 
    F_temp = np.zeros([m+col_left+col_right,n+row_top+row_bottom])
    F_temp[row_top:m+row_top, col_left:n+col_left] = 1.0
    F_temp[row_top:m+row_top, col_left:n+col_left] *= F
    I = np.zeros_like(F)
    
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
    
    print(row_top)
#    iterating over the original array while applying a horizontal kernel
#    with a max value filter
    for i in range(row_top,m+row_top):
        for j in range(col_left,n+col_left):
            
#            partitioning the array and computing the max value and storing
#            it in the output array
            snap = F_temp[i-row_top: i+row_bottom+1, j].copy()
            if snap[row_top] == np.amax(snap):
                I_horizontal[i-row_top,j-col_left] = snap[row_top]
#                print( (i-row_top, j-col_left))
#                print(snap[row_top])
#                print(np.amax(snap))
                
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
            

def image_thresholding(img, T):
    
    if 0 > T or T > 255:
        return 0
    
    B = np.zeros_like(img)
    
    for i in range(0,len(img)):
        for j in range(0,len(img[i])):
            
            if img[i,j] > T:
                B[i,j] = 1
    return B
