# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 19:52:16 2020

@author: Shubham
"""
import numpy as np
import cv2
"""
function stack list of images
images in images_1 and images_2 are stacked horizontally
set of images_1 and images_2 are stacked vertically
"""


def stacked_images(scale=1, images_1=[], images_2=[]):
    
    if len(images_2) != 0:  # for 2 set of images

        # add black images to small list
        if len(images_1) > len(images_2):
            for i in range((len(images_1) - len(images_2))):
                images_2.append(np.zeros_like(images_1[0]))

        elif len(images_1) < len(images_2):
            for i in range(len(images_2) - len(images_1)):
                images_1.append(np.zeros_like(images_2[0]))

        image_set = [images_1, images_2]
        
        # rescale images
        width, height = images_1[0].shape[0], images_1[0].shape[1]
        for i in range(len(image_set)):
            for j in range(len(image_set[0])):
                image_set[i][j] = cv2.resize(image_set[i][j], (int(height * scale), (int(width * scale))))
                
        # reshape image into 3 channels, if not 
        for i in range(len(image_set)):
            for j in range(len(image_set[0])):
                if len(image_set[i][j].shape) != 3:
                    image_set[i][j] = np.stack((image_set[i][j], image_set[i][j], image_set[i][j]), axis=2)
        
        hor_stacked_1 = np.hstack(images_1)
        hor_stacked_2 = np.hstack(images_2)
        stack = np.vstack((hor_stacked_1, hor_stacked_2))
        
    else:  # for 1 set of images
        
        # rescale images
        width, height = images_1[0].shape[0], images_1[0].shape[1]
        for i in range(len(images_1)):
            images_1[i] = cv2.resize(images_1[i], (int(height * scale), (int(width * scale))))
        
        # reshape image into 3 channels, if not 
        for i in range(len(images_1)):
            if len(images_1[i].shape) != 3:
                images_1[i] = np.stack((images_1[i], images_1[i], images_1[i]), axis=2)
        
        stack = np.hstack(images_1)
        
    return stack
