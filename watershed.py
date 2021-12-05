## watershed
##
## based on code from here: https://www.pyimagesearch.com/2015/11/02/watershed-opencv/
## uses watershed method for image segmentation
import sys
import cv2
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from scipy import ndimage
import numpy as np
import imutils



def main():

    # load the image and perform pyramid mean shift filtering
    # to aid the thresholding step
    path = './images/door.jpg'
    image = cv2.imread(path)
    image = cv2.resize(image, (2*256,2*256))
    shifted = cv2.pyrMeanShiftFiltering(image, 21, 51)
    
    # convert the mean shift image to grayscale, then apply
    # Otsu's thresholding
    gray = cv2.cvtColor(shifted, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255,
    	cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    # compute the exact Euclidean distance from every binary
    # pixel to the nearest zero pixel, then find peaks in this
    # distance map
    D = ndimage.distance_transform_edt(thresh)
    localMax = peak_local_max(D, indices=False, min_distance=20,
    	labels=thresh)
    # perform a connected component analysis on the local peaks,
    # using 8-connectivity, then appy the Watershed algorithm
    markers = ndimage.label(localMax, structure=np.ones((3, 3)))[0]
    labels = watershed(-D, markers, mask=thresh)
    print("[INFO] {} unique segments found".format(len(np.unique(labels)) - 1))
    
    # loop over the unique labels returned by the Watershed
    # algorithm
    for label in np.unique(labels):
        # if the label is zero, we are examining the 'background'
        # so simply ignore it
        if label == 0:
            continue
        # otherwise, allocate memory for the label region and draw
        # it on the mask
        mask = np.zeros(gray.shape, dtype="uint8")
        mask[labels == label] = 255
        # detect contours in the mask and grab the largest one
        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,	cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        c = max(cnts, key=cv2.contourArea)
        
        drawRect(c, image)
        
    while True:
        #cv2.imshow("Input", image)
        cv2.imshow("Thresh", thresh)
        cv2.imshow("Output", image)
        if cv2.waitKey(1) & (0xFF == 27 or 0xFF == ord('q')):break
 
def drawRect(c, img):
    rect = cv2.minAreaRect(c)
    #print(rect.size())
    size = rect[0]
    wdt = size[0]
    hgt = size[1]
    area = wdt*hgt;
    
    if True:#area > 0:
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        cv2.drawContours(img,[box],0,(0,0,255),2)
    
## runs the program and displays system errors    
if __name__ == "__main__":
    try:
        main()
    except:
        print("Unexpected error:", sys.exc_info()[0])
        cv2.destroyAllWindows()
