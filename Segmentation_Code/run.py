from __future__ import print_function

import os
import sys
import numpy as np
# import pandas as pd
import matplotlib.pyplot as plt 
import cv2
import time
from .helpers import implt, resize

# Import the preprocessing Module
from .words_Segmentation import *

def normalise_And_center(img):
    (m, s) = cv2.meanStdDev(img)
    m = m[0][0]
    s = s[0][0]
    img = img - m
    img = img / s if s>0 else img
    return img

def return_boxes(imageObject, see = False):
    try :
        if imageObject is not None:
            boxesFound = detection(imageObject, see, join = False)
            print(type(boxesFound))
            return boxesFound
    
    except Exception as E:
        print(E)
        pass

def otsu_thresholding(imageObject):


    img_gray = cv2.cvtColor(imageObject, cv2.COLOR_BGR2GRAY)
    blur = cv2.boxFilter(img_gray, -1, (3, 3))

    _ ,th3 = cv2.threshold(blur,0,255,cv2.THRESH_OTSU)

    return th3

def extract_boxes(imageObject, boxes, apply_otsu_on_boxes, to_save_location, filename):
    cnt = 0
    for i, (startX, startY, endX, endY) in enumerate(boxes):
        
        padding = -1 #Padding to be 5 pixels
        startX = int(startX - padding)  
        startY = int(startY - padding) 
        endX = int(endX + padding) 
        endY = int(endY + padding)

        
        newBox = imageObject[startY:endY, startX:endX]
        if(apply_otsu_on_boxes):
            newBox = otsu_thresholding(newBox)
        # newBox = normalise_And_center(newBox)
        # implt(newBox)
        
        
        if not os.path.isdir(f"{to_save_location}{str(filename.split('.')[0])}"):
            print('NOT FOUND')
            os.mkdir(f"{to_save_location}{str(filename.split('.')[0])}")
        print(newBox)
        
        cv2.imwrite(f'{to_save_location}/{str(filename.split(".")[0])}/Box{cnt}.jpg', newBox)
        print(f'{to_save_location}{str(filename.split(".")[0])}/Box{cnt}.jpg')
        cnt += 1
        
def get_input_output_folder_path():
    # By default 
    # The input images will be taken from ../Input_Images Folder
    # And then the segmented images will be put into ../Word_Segmented_Images Folder
    # You can also change this
    return ("Input_Images/", "Word_Segmented_Images/")

def run(image_filename):
    # Supports .jpg, .jpeg, .tiff
    # In other words all the file-types that opencv supports

    input_images_folder, output_images_folder = get_input_output_folder_path()
    assert(input_images_folder[-1] == "/"  and output_images_folder[-1] == "/")

    originalImage = cv2.imread(input_images_folder + image_filename)
    boxes  = return_boxes(originalImage, see = False)
    newImage = cv2.imread(input_images_folder + image_filename, cv2.IMREAD_GRAYSCALE)
    newImage = resize(newImage, 2000)

    # Lets do otsu threshold on this image
    # newImage = otsu_thresholding(newImage)
    # Extract the boxes and save the files 
    extract_boxes(newImage, boxes, filename = image_filename, \
         to_save_location = output_images_folder, apply_otsu_on_boxes= False)





if __name__ == "__main__":
    # IMG_PATH = "../data/images/Paras_Image/IMG_20200228_122609~2.jpg"
    # IMG_PATH = "../data/images/doctor_Written_Image.jpg"
    #IMG_PATH = "./CUT_LINES/cut_horizontal_LINE2.jpg"
    IMG_PATH = sys.argv[1]    
    print(IMG_PATH)

    # LOad the image
    originalImage = cv2.imread(IMG_PATH)
    print(originalImage)
    boxes  = return_boxes(originalImage, see = True)

    print(boxes)

    newImage = cv2.imread(IMG_PATH, cv2.IMREAD_GRAYSCALE)
    newImage = resize(newImage, 2000)

    # Lets do otsu threshold on this image
    # newImage = otsu_thresholding(newImage)
    extract_boxes(newImage, boxes, apply_otsu_on_boxes= False)
