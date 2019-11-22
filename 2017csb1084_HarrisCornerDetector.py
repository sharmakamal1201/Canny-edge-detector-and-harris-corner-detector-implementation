# Harris corner detector
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy.ndimage import gaussian_filter
from scipy.ndimage import filters
from scipy.ndimage import convolve
import operator


# my 5*5 sobel filter
def myFilter(img, flag):
    if(flag == 1):
        Fx = np.array([[-1,0,+1], [-2,0,+2], [-1,0,+1]])
        out = convolve(img, Fx)
    else:
        Fy = np.array([[-1,-2,-1], [0,0,0], [+1,+2,+1]])
        out = convolve(img, Fy)    
    return out
        
def HarrisCorner(img_cpy, gray_img, Harris_const, window):
    Fx = np.zeros(gray_img.shape)
    Fx = myFilter(gray_img,1)
    Fy = np.zeros(gray_img.shape)
    Fy = myFilter(gray_img,0) 
    Fxx = Fx**2
    Fyy = Fy**2
    Fxy = Fx*Fy

    height, width, channels = img_cpy.shape
    offset = int(window/2)
    cornerList = [] 
    cornerList2 = []
    Th = 4
    
    for y in range(offset, height-offset):
        for x in range(offset, width-offset):
            Sxx = np.sum(Fxx[y-offset:y+offset+1, x-offset:x+offset+1])
            Syy = np.sum(Fyy[y-offset:y+offset+1, x-offset:x+offset+1])
            Sxy = np.sum(Fxy[y-offset:y+offset+1, x-offset:x+offset+1])
            det = (Sxx * Syy) - (Sxy**2)
            trace = Sxx + Syy
            r = det - Harris_const * (trace**2)
            if r > Th:
                cornerList.append([x, y, r])
                
    print(len(cornerList))
    cornerList.sort(key=operator.itemgetter(2), reverse=True)
    cornerList2 = cornerList
    for i in range (len(cornerList)):
        X1, Y1, R1 = cornerList[i]
        arr1 = [0,0,1,1,1,-1,-1,-1]
        arr2 = [1,-1,0,1,-1,0,1,-1]
        for j in range(8):
            for z in range(len(cornerList)):
                if(cornerList[z][0]==X1+arr1[j] and cornerList[z][1]==Y1+arr2[j] and cornerList[z][2]<R1):
                    cornerList2[z][2] = 0
        
    cornerList2.sort(key=operator.itemgetter(2), reverse=True)
    mini = min(100,len(cornerList2))
    print(mini)
    for i in range (mini):
        X1, Y1, R1 = cornerList2[i]
        print(i,' --- ',R1)
        if(R1<=0):
            break;
        img_cpy[Y1,X1] = [255,0,0]
    
    plt.imshow(img_cpy)
    plt.show()
    return img_cpy





    
img_name = 'plane'
img = mpimg.imread('data/'+img_name+'.bmp')
img_cpy = img.copy()

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

gray_img = rgb2gray(img) 
gray_img = gray_img/np.max(gray_img)
img_cpy=HarrisCorner(img_cpy, gray_img, 0.04, 3)
mpimg.imsave('Result_harris/'+img_name+'.bmp',img_cpy)