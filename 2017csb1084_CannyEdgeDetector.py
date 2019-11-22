#canny edge detector
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy.ndimage import gaussian_filter
from scipy.ndimage import filters
from scipy.ndimage import convolve

def show_img(img_name):
    plt.imshow(img_name, cmap = plt.get_cmap('gray'))
    plt.show()

#read input image
img_name = 'plane'
img = mpimg.imread('data/'+img_name+'.bmp')
#plot input image
imgplot = plt.imshow(img)
plt.show()

#convert to grayscale and plot
def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

gray_img = rgb2gray(img)    
show_img(gray_img)

# smoothng with gaussian filter
gray_img_blurred = gaussian_filter(gray_img, sigma = 1) # sigma = 1 assumed or given in question
show_img(gray_img_blurred)
mpimg.imsave('Result_canny/Smoothen_images/'+img_name+'.bmp',gray_img_blurred, cmap = plt.get_cmap('gray'))

# my 5*5 sobel filter
def myFilter(img, flag):
    if(flag == 1):
        Fx = np.array([[-1,-2,0,+2,+1], [-4,-8,0,+8,+4], [-6,-12,0,+12,+6], [-4,-8,0,+8,+4], [-1,-2,0,+2,+1]])
        out = convolve(img, Fx)
    else:
        Fy = np.array([[-1,-4,-6,-4,-1], [-2,-8,-12,-8,-2], [0,0,0,0,0], [+2,+8,+12,+8,+2], [+1,+4,+6,+4,+1]])
        out = convolve(img, Fy)    
    return out

# x and y components of image gradient
Fx = np.zeros(gray_img_blurred.shape)
Fx = myFilter(gray_img_blurred,1)
#Fx = np.zeros(gray_img.shape)
#Fx = myFilter(gray_img,1)
#filters.sobel(gray_img_blurred,1,Fx)
show_img(Fx)
mpimg.imsave('Result_canny/Sobel_filter_results/'+img_name+'_x.bmp',Fx, cmap = plt.get_cmap('gray'))


Fy = np.zeros(gray_img_blurred.shape)
Fy = myFilter(gray_img_blurred,0)
#filters.sobel(gray_img_blurred,0,Fy)
#Fx = np.zeros(gray_img.shape)
#Fx = myFilter(gray_img,0)
show_img(Fy)
mpimg.imsave('Result_canny/Sobel_filter_results/'+img_name+'_y.bmp',Fy, cmap = plt.get_cmap('gray'))


# magnitude and direction of gradient
Magnitude = np.sqrt(Fx**2 + Fy**2)
Magnitude = Magnitude / np.max(Magnitude) # normalise
Gradient_direction = np.degrees(np.arctan2(Fy,Fx))
show_img(Magnitude)
mpimg.imsave('Result_canny/Gradient_magnitude/'+img_name+'.bmp',Magnitude, cmap = plt.get_cmap('gray'))
mpimg.imsave('Result_canny/Gradient_direction/'+img_name+'.bmp',Gradient_direction, cmap = plt.get_cmap('gray'))


# NMS
#idea from https://towardsdatascience.com/canny-edge-detection-step-by-step-in-python-computer-vision-b49c3a2d8123
def NMS(Mag, Dir):
    pi = 180.0
    res_img = np.zeros(Mag.shape)
    M, N = Mag.shape
    for i in range(1, M-1):
        for j in range(1, N-1):
            D = Dir[i,j]
            if(D<0):
                D = D + 180
            first = 1
            second = 1
            if((D >= 0 and D <= pi/8.0) or (D >= (7*pi)/8.0 and D <= pi)):
                first = Mag[i,j+1]
                second = Mag[i,j-1]
            elif((D >= pi/8.0 and D <= (3*pi)/8.0)):
                first = Mag[i+1,j+1]
                second = Mag[i-1,j-1]
            elif((D >= (3*pi)/8.0 and D <= (5*pi)/8.0)):
                first = Mag[i+1,j]
                second = Mag[i-1,j]
            elif((D >= (5*pi)/8.0 and D <= (7*pi)/8.0)):
                first = Mag[i+1,j-1]
                second = Mag[i-1,j+1]
            if((Mag[i,j]>first) and (Mag[i,j]>second)):
                res_img[i,j] = Mag[i,j]
    return res_img

# plot image after non-maximal suppression
img_after_nms = NMS(Magnitude, Gradient_direction)
show_img(img_after_nms)
mpimg.imsave('Result_canny/NMS_result/'+img_name+'.bmp',img_after_nms, cmap = plt.get_cmap('gray'))


def isPossible(cpy,i,j,Th):
    cur = cpy[i,j]
    top = cpy[i-1,j]
    btm = cpy[i+1,j]
    lft = cpy[i,j-1]
    rgt = cpy[i,j+1]
    toplft = cpy[i-1,j-1]
    toprgt = cpy[i-1,j+1]
    btmlft = cpy[i+1,j-1]
    btmrgt = cpy[i+1,j+1]
    return ((cur>Th) or (top>Th) or (btm>Th) or (lft>Th) or (rgt>Th) or (toplft>Th) or (toprgt>Th) or (btmlft>Th) or (btmrgt>Th))

def dfs(cpy, Tl, Th):
    total = 1
    prev_total = 0
    M, N = cpy.shape
    while(total != prev_total):
        prev_total = total
        total = 0
        for i in range(1,M-1):
            for j in range(1,N-1):
                if(cpy[i,j] < Tl):
                    cpy[i,j] = 0
                elif(isPossible(cpy,i,j,Th)):
                    cpy[i,j] = 1
                    total = total + 1
    for i in range(1,M-1):
        for j in range(1,N-1):
            if(cpy[i,j]!=1):
                cpy[i,j]=0
    
    return cpy

def Hys_threshold(img, Th_ratio, Tl_ratio):
    cpy = np.copy(img)
    Th = np.max(cpy) * Th_ratio
    Tl = Th * Tl_ratio    
    return dfs(cpy,Tl,Th)
Output_img = Hys_threshold(img_after_nms, 0.28, 0.2)
show_img(Output_img)
print(Output_img.shape)
#mpimg.imsave('Result_canny/Final_output/'+img_name+'.bmp',Output_img, cmap = plt.get_cmap('gray'))