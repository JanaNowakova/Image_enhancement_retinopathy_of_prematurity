# -*- coding: utf-8 -*-
"""
Created on Fri Apr 29 18:47:41 2022

@author: 
    
    Testing code for image enhancement for infant childern retinal iamges (fundus images)
    Most of methods are implemented in Python 3.9x and OpenCV library
    except:
        Mirnet available at https://github.com/swz30/MIRNet
        AGCCPF available at https://pypi.org/project/image-enhancement/
        
    
"""
# imports 
import os
import shutil
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import cv2
import numpy as np


# Where to save the figures
PROJECT_ROOT_DIR = "."
PROCESSED = "processed"
IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, "images", PROCESSED)
os.makedirs(IMAGES_PATH, exist_ok=True)

def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = os.path.join(IMAGES_PATH, fig_id + "." + fig_extension)
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)
    





FILE = 'image1.jpg'

# dg 05 ROP 2 
#FILE = '051_F_GA24_BW600_PA42_DG4_PF0_D1_S03_7.jpg'

# dark image 
#FILE = '006_F_GA40_BW3200_PA44_DG13_PF0_D1_S02_1.jpg'



""" image transformation """




""" 0: RGB channels histogram visualization"""

def plot_RGB_hist(img, img_path=None):
    # visualize RGB color histogram
    # load image
    if img_path:
        colormap1 = cv2.imread(img_path)
    else:
        colormap1 = cv2.imread(os.path.join(os.getcwd(),img))
    # BGR2RG
    colormap1=cv2.cvtColor(colormap1, cv2.COLOR_BGR2RGB)
    # plt.imshow(colormap1) # plot original image if necessary
    # get channels
    chans=cv2.split(colormap1)
    colors=("b", "g", "r")
    plt.figure()
    plt.title("Color histogram")
    plt.xlabel("Bins")
    plt.ylabel("Number of pixels")
    for (chan, c) in zip(chans, colors):
        hist=cv2.calcHist([chan], [0], None, [256], [0,256])
        plt.plot(hist, color=c)
        plt.xlim([0,256])
    plt.show()
    
    
plot_RGB_hist(FILE)




""" 00: GET image in different  channels """ 

def split_RGB_channels(file, file_path=None, gray=False):
    if file_path:
        img_RGB = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
    else:
        img_RGB = cv2.imread(os.path.join(os.getcwd(),file), cv2.IMREAD_UNCHANGED)
   
    
    r, g, b = cv2.split(img_RGB) 
        
    if gray: 
        # show channels in gray colors
        plt.figure(figsize=(20,5))
        plt.subplot(151); plt.imshow(r,cmap='gray'); plt.title('Red Channel')
        plt.subplot(152); plt.imshow(g,cmap='gray'); plt.title('Green Channel')    
        plt.subplot(153); plt.imshow(b,cmap='gray'); plt.title('Blue Channel')
    else:
        zero_ch = np.zeros(img_RGB.shape[0:2], dtype="uint8")
        blue_img = cv2.merge([b, zero_ch, zero_ch])
        green_img = cv2.merge([zero_ch, g, zero_ch])
        red_img = cv2.merge([zero_ch, zero_ch, r])
        
        plt.figure(figsize=(20,5))
        plt.subplot(151); plt.imshow(red_img); plt.title('Red Channel')
        plt.subplot(152); plt.imshow(green_img); plt.title('Green Channel')    
        plt.subplot(153); plt.imshow(blue_img); plt.title('Blue Channel')
        
    r, g, b = cv2.split(img_RGB)  
    # merge the r,g, and b channels int BGR image
    imgMerged = cv2.merge((b,g,r))
    plt.subplot(154); plt.imshow(imgMerged[:,:,::-1]); plt.title('Merged BGR')
    imgMerged_RGB = cv2.merge((r,g,b))
    # change BGR to RGB img[...,::-1]
    plt.subplot(155); plt.imshow(imgMerged_RGB[:,:,::-1]); plt.title('Original')
    
split_RGB_channels(FILE)    





""" 1: FILTER: HISTOGRAM Equalization -  gray """


def hist_equalization_gray(file, file_path=None, show_hist=False, clipLimit = 2.0):
    # CLAHE histogram on Grayscale image
    
    if file_path:
        img = cv2.imread(file_path, 0)
    else:
        img = cv2.imread(os.path.join(os.getcwd(),file), 0)
        
    # create a CLAHE object (Arguments are optional).
    # clipLimit	= Threshold for contrast limiting. 
    clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=(8,8))
    cl1 = clahe.apply(img)
    if show_hist: # alternatively show histograms
        plt.figure(figsize=(10,7))
        plt.subplot(221); plt.imshow(img); plt.title('Original')
        plt.subplot(222); plt.imshow(cl1); plt.title('Clahe') 
        
        hist,bins = np.histogram(img.flatten(),256,[0,256])
        cdf = hist.cumsum()
        cdf_normalized = cdf * float(hist.max()) / cdf.max()
        plt.subplot(223) 
        plt.plot(cdf_normalized, color = 'b')
        plt.hist(img.flatten(),256,[0,256], color = 'r')
        plt.xlim([0,256])
        plt.legend(('cdf','histogram'), loc = 'upper left')
        plt.title('Original histogram')
        

        hist,bins = np.histogram(cl1.flatten(),256,[0,256])
        cdf = hist.cumsum()
        cdf_normalized = cdf * float(hist.max()) / cdf.max()
        plt.subplot(224) 
        plt.plot(cdf_normalized, color = 'b')
        plt.hist(img.flatten(),256,[0,256], color = 'r')
        plt.xlim([0,256])
        plt.legend(('cdf','histogram'), loc = 'upper left')
        plt.title('Clahe histogram') 
        plt.show()
    else:
        # visualization is in jet color scheme
        plt.figure(figsize=(10,5))
        plt.subplot(121); plt.imshow(img); plt.title('Original')
        plt.subplot(122); plt.imshow(cl1); plt.title('Clahe')
        #save_fig('Clahe')
        plt.show()
          
    
hist_equalization_gray(FILE,show_hist=False, clipLimit=2.2)   



""" 2: FILTER: HISTOGRAM Equalization -  color """

def hist_equalization_color(file, file_path=None, clipLimit = 2.0, tileGridSize=(8,8)):
    # application of CLAHE algoritm on RGB image
    if file_path:
        img_RGB = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
    else:
        img_RGB = cv2.imread(os.path.join(os.getcwd(),file), cv2.IMREAD_UNCHANGED)
    
    # to HSV format
    hsv_img = cv2.cvtColor(img_RGB, cv2.COLOR_BGR2HSV)
        
    h, s, v = hsv_img[:,:,0], hsv_img[:,:,1], hsv_img[:,:,2]
    # create a CLAHE object (Arguments are optional).
    # clipLimit	= Threshold for contrast limiting. 
    # apply CLAHE on V part
    clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
    v = clahe.apply(v)
    # complete image
    hsv_img = np.dstack((h,s,v))
    # visualise modified and original image 
    rgb = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2RGB)
    img_RGB = cv2.cvtColor(img_RGB, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(10,5))
    plt.subplot(121); plt.imshow(img_RGB); plt.title('Original')
    plt.subplot(122); plt.imshow(rgb); plt.title('Clahe') 
    plt.show()
    

hist_equalization_color(FILE, clipLimit=1.8, tileGridSize=(20,20)) 




""" 3: FILTER: HISTOGRAM Equalization -  color with mask  """ # does not improve visualization 

def hist_equalization_green(file, file_path=None, clipLimit = 2.0, tileGridSize=(8,8)):
    # application of CLAHE algoritm on RGB image  or green part (possibly) 
    # with mask limiting some parts of image
    # it could produce better results on specific images but not in general
    if file_path:
        img_RGB = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
    else:
        img_RGB = cv2.imread(os.path.join(os.getcwd(),file), cv2.IMREAD_UNCHANGED)
        
    hsv_img = cv2.cvtColor(img_RGB, cv2.COLOR_BGR2HSV)
    
    ## mask of green (36,25,25) ~ (86, 255,255)
    # mask = cv2.inRange(hsv, (36, 25, 25), (86, 255,255))
    mask = cv2.inRange(hsv_img, (36, 25, 25), (70, 255,255))
    imask = mask>0
    # green = np.zeros_like(hsv_img, np.uint8)
    hsv_img[imask] = hsv_img[imask]
        
    h, s, v = hsv_img[:,:,0], hsv_img[:,:,1], hsv_img[:,:,2]
    # create a CLAHE object (Arguments are optional).
    # clipLimit	= Threshold for contrast limiting. 
    clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
    v = clahe.apply(v)
    hsv_img = np.dstack((h,s,v))
    rgb = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2RGB)
    img_RGB = cv2.cvtColor(img_RGB, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(10,5))
    plt.subplot(121); plt.imshow(img_RGB); plt.title('Original')
    plt.subplot(122); plt.imshow(rgb); plt.title('Clahe Mask Green') 
    plt.show()
    

# hist_equalization_green(FILE, clipLimit=1.8, tileGridSize=(20,20)) 





""" 4: FILTER: HISTOGRAM Equalization -  green channel """

def hist_equalization_green_channel(file, file_path=None, show_hist=False, clipLimit = 2.0):
    # CLAHE histogram on Gren part of image
    if file_path:
        img_RGB = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
    else:
        img_RGB = cv2.imread(os.path.join(os.getcwd(),file), cv2.IMREAD_UNCHANGED)
   
        
    r, g, b = cv2.split(img_RGB)
    zero_ch = np.zeros(img_RGB.shape[0:2], dtype="uint8")
    green_img = cv2.merge([zero_ch, g, zero_ch])
        
    # create a CLAHE object (Arguments are optional).
    # clipLimit	= Threshold for contrast limiting. 
    clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=(8,8))
    cl1 = clahe.apply(g)
    cl1 = cv2.merge([zero_ch, cl1, zero_ch])
    
    if show_hist:
        plt.figure(figsize=(10,7))
        plt.subplot(221); plt.imshow(green_img); plt.title('Green Channel')  
        plt.subplot(222); plt.imshow(cl1); plt.title('CLAHE on green channel') 
        
        hist,bins = np.histogram(g.flatten(),256,[0,256])
        cdf = hist.cumsum()
        cdf_normalized = cdf * float(hist.max()) / cdf.max()
        plt.subplot(223) 
        plt.plot(cdf_normalized, color = 'b')
        plt.hist(g.flatten(),256,[0,256], color = 'r')
        plt.xlim([0,256])
        plt.legend(('cdf','histogram'), loc = 'upper left')
        plt.title('Original histogram')
        

        hist,bins = np.histogram(cl1.flatten(),256,[0,256])
        cdf = hist.cumsum()
        cdf_normalized = cdf * float(hist.max()) / cdf.max()
        plt.subplot(224) 
        plt.plot(cdf_normalized, color = 'b')
        plt.hist(g.flatten(),256,[0,256], color = 'r')
        plt.xlim([0,256])
        plt.legend(('cdf','histogram'), loc = 'upper left')
        plt.title('Clahe histogram') 
        plt.show()
    else:
        plt.figure(figsize=(10,5))
        plt.subplot(121); plt.imshow(green_img); plt.title('Original')
        plt.subplot(122); plt.imshow(cl1,cmap='gray'); plt.title('Clahe')
        #save_fig('Clahe')
        plt.show()    
    
    
hist_equalization_green_channel(FILE,show_hist=False, clipLimit=2.2)   



    


""" 5: FILTER : CROP IMAGE and Gaussian Blur application """

def crop_image_from_gray(img,tol=10):
    if img.ndim ==2:
        mask = img>tol
        # plt.matshow((mask.astype(int)))
        # crop 
        return img[np.ix_(mask.any(1),mask.any(0))]
    elif img.ndim==3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mask = gray_img>tol
        
        check_black = img[:,:,0][np.ix_(mask.any(1),mask.any(0))].shape[0]
        if (check_black == 0): # image is too dark decrease tolerance 
            return img 
        else:
            img1=img[:,:,0][np.ix_(mask.any(1),mask.any(0))]
            img2=img[:,:,1][np.ix_(mask.any(1),mask.any(0))]
            img3=img[:,:,2][np.ix_(mask.any(1),mask.any(0))]
    #         print(img1.shape,img2.shape,img3.shape)
            img = np.stack([img1,img2,img3],axis=-1)
    #         print(img.shape)
        return img

     
    
def circle_crop(img, sigmaX = 30):   
    # circular crop around image centre with application of Gaussian blur
    img = crop_image_from_gray(img)    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    height, width, depth = img.shape    
    
    x = int(width/2)
    y = int(height/2)
    r = np.amin((x,y))
    
    circle_img = np.zeros((height, width), np.uint8)
    cv2.circle(circle_img, (x,y), int(r), 1, thickness=-1)
    # remove everyhing behind horizont of circle 
    img = cv2.bitwise_and(img, img, mask=circle_img)
    img = crop_image_from_gray(img)
    # add gausian blur
    # img  = src1*alpha + src2*beta + gamma
    img=cv2.addWeighted(img,5, cv2.GaussianBlur( img , (0,0) , sigmaX) ,-5 ,128)
    return img 

def circle_crop_preprocess(file, file_path=None, cir_crop=True, sigmaX=35):
    if file_path:
        img = cv2.imread(file_path, cv2.IMREAD_COLOR)
    else:
        img = cv2.imread(os.path.join(os.getcwd(),file), cv2.IMREAD_COLOR)
    
    img = crop_image_from_gray(img)
    if cir_crop: # if False, just crop black part of retinal image
        img = circle_crop(img, sigmaX) 
    plt.figure(figsize=(5,5))
    plt.imshow(img)
    # save_fig('1_Circle_Crop')
    plt.show()
    
    
    

circle_crop_preprocess(FILE, file_path=None, cir_crop=True, sigmaX=15)   


""" Note: another alternative is 

https://github.com/mcamila777/DL-to-retina-images/blob/master/notebooks/Mean_Std_calculation.ipynb

 """
    
 
""" 6: FILTER : Tresholding to gray image """   
    
def tresholding(file, file_path=None):
    # application of tresholding methods to retinal image
    if file_path:
        img= cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
    else:
        img = cv2.imread(os.path.join(os.getcwd(), file), cv2.IMREAD_UNCHANGED)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    x, y, _ = img.shape
    scale = round( max(x,y)*0.5) 
    # apply morphology
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE , (scale,scale))
    #opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    morph = cv2.morphologyEx(img_gray, cv2.MORPH_OPEN, kernel)
    # divide gray by morphology image
    division = cv2.divide(img_gray, morph, scale=100)


    img_blur = cv2.medianBlur(img,5)
   
    # threshold
    threshold = cv2.adaptiveThreshold(img_gray,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,11,2)
    threshold2 = cv2.threshold(img_gray, 0, 255, cv2.THRESH_OTSU )[1]
    
    titles = ['Original Image', 'Division',
                'Adaptive Mean Thresholding', 'Otsu on division Thresholding']
    images = [img, division, threshold, threshold2]
    
    
    plt.figure(figsize=(10,10))
    for i in range(4):
        plt.subplot(2,2,i+1),plt.imshow(images[i],'gray')
        plt.title(titles[i])
        plt.xticks([]),plt.yticks([])
    plt.show()  
    
    
tresholding(FILE)   
    



""" 5: FILTER : Standartize brightness """

from numpy.linalg import norm

# nomr computation motivated by
# https://stackoverflow.com/questions/14243472/estimate-brightness-of-an-image-opencv
# m = np.arange(12).reshape(3,2,2) ; np.sum(m,axis=2) # for norm computation understanding
def get_brightness(img):
     # input colored RGB BGR image
    if len(img.shape) == 3:
        #  brightness by euclidean norm # plt.imshow(norm(img, axis=2))
        return np.average(norm(img, axis=2)) / np.sqrt(3)
    else:
        # Grayscale
        return np.average(img)



def adjust_brightness(file, file_path=None, min_brightness = 80):   
    if file_path:
        img = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
    else:
        img = cv2.imread(os.path.join(os.getcwd(), file), cv2.IMREAD_UNCHANGED)
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    brightness = get_brightness(img)
    print('Brightness:', brightness)
    # ratio = brightness / min_brightness
    if brightness >= min_brightness: # brightness is bright enought
        img_adj = img
    else:
        # compute brightness adjustment
        img_adj = cv2.convertScaleAbs(img, alpha = min_brightness / brightness, 
                                      beta =  (min_brightness - brightness)  )
    
    plt.figure(figsize=(10,5))
    plt.subplot(121); plt.imshow(img); plt.title('Original')
    plt.subplot(122); plt.imshow(img_adj); plt.title('Adjusted')
    #save_fig('Clahe')
    plt.show()
    
    #plt.figure(figsize=(5,5))
    #plt.imshow(img_adj)
    #save_fig('2_Adjusted_my')
    plt.show()
    
 
FILE = 'image2.jpg'    
adjust_brightness(FILE, min_brightness=120) 
    
    
    
    

    
    
    