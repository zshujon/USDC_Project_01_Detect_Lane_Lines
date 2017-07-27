# -*- coding: utf-8 -*-
"""
Created on Tue Jul 25 12:47:57 2017

@author: ZSHUJON
"""

import matplotlib.pyplot as plt
import numpy as np
import cv2

from helper import TRANSFORM_POINTS_00, TRANSFORM_POINTS_01, TRANSFORM_POINTS_02, TRANSFORM_POINTS_03

def imread(path):
    img = cv2.imread(path)
    if len(img.shape) > 2:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img
        
def imshow(img, cmap=None, norm=None, aspect=None, interpolation=None, alpha=None, vmin=None, vmax=None, origin=None, extent=None, shape=None, filternorm=1, filterrad=4.0, imlim=None, resample=None, url=None, hold=None, data=None, **kwargs):
    if len(img.shape) > 2:
        plt.imshow(img, cmap); 
        plt.show()
    else:
        plt.imshow(img, cmap='gray')
        plt.show()

def rgb_to_(img, what=None):
    if what=='gray':
        return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    elif what=='hsv':
        return cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    elif what=='hls':
        return cv2.cvtColor(img, cv2.COLOR_RGB2HLS)

def edge(img, edg):
    sigma = 0.33
    if edg=='canny':
        median_of_img = np.median(img)
        lower_thresh = int(max(0, (1.0 - sigma) * median_of_img))
        upper_thresh = int(max(0, (1.0 + sigma) * median_of_img))
        return cv2.Canny(img, lower_thresh, upper_thresh)

def gaussian_blur(img):
    return cv2.GaussianBlur(img, (3, 3), 0)

def four_point_prespective_transform(img):
    pts1 = np.float32([TRANSFORM_POINTS_00,TRANSFORM_POINTS_01,TRANSFORM_POINTS_02,TRANSFORM_POINTS_03])
    pts2 = np.float32([[0,650],[300,0],[900,0],[img.shape[1],650]])
    
    M = cv2.getPerspectiveTransform(pts1,pts2)
    transformed_img = cv2.warpPerspective(img,M,(img.shape[1],img.shape[0]), flags=cv2.INTER_LINEAR)
    return transformed_img

def hough_lines(img):
    theta = np.pi/180     
    rho = 1
    threshold = 50
    min_line_len = 10
    max_line_gap = 150
    
    hough_lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)

    return hough_lines

def white_yellow_portion(img):
    hsv = rgb_to_(img, 'hsv')
    # white color mask
    lower = np.uint8([0,200,0])
    upper = np.uint8([255,255,255])
    white_mask = cv2.inRange(hsv, lower, upper)
    #imshow(white_mask)
    # yellow color mask
    lower = np.uint8([10,0,100])
    upper = np.uint8([40,255,255])
    yellow_mask = cv2.inRange(hsv, lower, upper)
    #imshow(yellow_mask)
    # combine the mask
    mask = cv2.bitwise_or(white_mask, yellow_mask)
    return cv2.bitwise_and(img, img, mask = mask)