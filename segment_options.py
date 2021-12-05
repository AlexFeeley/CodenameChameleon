## segment_options
##
## can use k-means clustering, contour detection, or threshold based segmentation 
## based off of https://machinelearningknowledge.ai/image-segmentation-in-python-opencv/

import sys
import matplotlib as plt
from skimage.filters import threshold_otsu
import numpy as np
import cv2

def main():
    path = './images/water_coins.jpg'
    img = cv2.imread(path)
    # maximum desired dimension
    # max_des_dim = 256
    # img = scale(img, max_des_dim)

    # easier to scale image to a square for now
    img = cv2.resize(img,(256,256))
    
    # pick which method to use and uncomment

    #img = cluster(img)
    #img =contour_detect(img)
    img = thresh_seg(img)
    display(img)
    
    
    
## scale
## resizes image to desired max dimension
## TODO may want to make it so all input images are forced to same exact dimensions
def scale(img, max_des_dim):
    # get max dimension, width or height, for scaling purposes
    width = img.shape[1]
    hgt = img.shape[0]
    max_dim = max(width, hgt)
    
    # resize image
    width = int(width/max_dim*max_des_dim)
    hgt = int(hgt/max_dim*max_des_dim)
    dim = (width, hgt)
    img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    
    return img



## cluster
## apply k-means clustering to image and return resulting image
def cluster(img):
    # preprocess image
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    twoDimage = img.reshape((-1,3))
    twoDimage = np.float32(twoDimage)
    
    
    # define clustering criteria
    # K = number of clusters a pixel can belong to
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = 3
    attempts=10
    
    # apply k-means
    # determine which cluster each part of the picture belongs to
    ret,label,center=cv2.kmeans(twoDimage,K,None,criteria,attempts,cv2.KMEANS_PP_CENTERS)
    center = np.uint8(center)
    res = center[label.flatten()]
    result_image = res.reshape((img.shape))
    
    return result_image



## contour_detect
## detects contours in image and returns a segmented image contours
def contour_detect(img):
    
    # convert to grayscale, threshold, and make detected edges clear
    gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    _,thresh = cv2.threshold(gray, np.mean(gray), 255, cv2.THRESH_BINARY_INV)
    edges = cv2.dilate(cv2.Canny(thresh,0,255),None)

    #detect and draw contours
    cnt = sorted(cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[-2], key=cv2.contourArea)[-1]
    mask = np.zeros((256,256), np.uint8)
    masked = cv2.drawContours(mask, [cnt],-1, 255, -1)
    
    #segment regions
    dst = cv2.bitwise_and(img, img, mask=mask)
    segmented = cv2.cvtColor(dst, cv2.COLOR_BGR2RGB)
    return segmented



## thresh_seg
## segments image using thresholding
def thresh_seg(img):
    # preprocess- convert to RGB
    img_rgb=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img_gray=cv2.cvtColor(img_rgb,cv2.COLOR_RGB2GRAY)
    # filter image using mask
    thresh = threshold_otsu(img_gray)
    img_otsu  = img_gray < thresh
    filtered = filter_image(img, img_otsu) 
    return filtered



## filter_image
## multiplies the maks with RGB channels and concats to form a normal image
def filter_image(image, mask):
    r = image[:,:,0] * mask
    g = image[:,:,1] * mask
    b = image[:,:,2] * mask
    return np.dstack([r,g,b])

 

## display
## displays the given image until ctrl+c 
## TODO would be nice to have a better way to display images but I think openCV's method is actually just broken
def display(img):
    while True:
        cv2.namedWindow('Result', cv2.WINDOW_NORMAL)
        cv2.imshow('Result', img)
        if cv2.waitKey(1) & (0xFF == 27 or 0xFF == ord('q')):break
    
    
    
## runs the program and displays system errors    
if __name__ == "__main__":
    try:
        main()
    except:
        print("Unexpected error:", sys.exc_info()[0])
        cv2.destroyAllWindows()
