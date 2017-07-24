# -*- coding: utf-8 -*-
"""
Created on Sun Jul 23 12:09:42 2017

@author: ASUS
"""

import matplotlib.pyplot as plt
import numpy as np
import cv2
from time import sleep
import imageio
#%%

#==============================================================================
# TRANSFORM_POINTS_00 = [100,500]
# TRANSFORM_POINTS_01 = [360,350]
# TRANSFORM_POINTS_02 = [600,350]
# TRANSFORM_POINTS_03 = [920,500]
#==============================================================================

TRANSFORM_POINTS_00 = [200,650]
TRANSFORM_POINTS_01 = [500,480]
TRANSFORM_POINTS_02 = [780,480]
TRANSFORM_POINTS_03 = [1150,650]

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

def points_to_transform(img):
    pts = np.array([TRANSFORM_POINTS_00, TRANSFORM_POINTS_01, TRANSFORM_POINTS_02, TRANSFORM_POINTS_03], np.int32)
    ptss = pts.reshape((-1,1,2))
    transformed = cv2.polylines(img.copy(),[ptss],True,(0,255,255), 2)
    return transformed
    
def four_point_prespective_transform(img):
    pts1 = np.float32([TRANSFORM_POINTS_01,TRANSFORM_POINTS_02,TRANSFORM_POINTS_00,TRANSFORM_POINTS_03])
    pts2 = np.float32([[0,0],[400,0],[0,400],[400,400]])
    
    M = cv2.getPerspectiveTransform(pts1,pts2)
    transformed_img = cv2.warpPerspective(img,M,(400,400), flags=cv2.INTER_LINEAR)
    return transformed_img

def hough_lines(img):
    theta = np.pi/180     
    rho = 1
    threshold = 50
    min_line_len = 10
    max_line_gap = 150
    
    hough_lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)

    return hough_lines

def draw_lines(img, lines):    
    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(img,(x1,y1),(x2,y2),(0,255,140),5)
    return img

def line_detect(img):
    transformed_image = four_point_prespective_transform(img)
    road_image_gray = rgb_to_(transformed_image, 'gray')
    road_image_blur = gaussian_blur(road_image_gray)
    road_image_blur_edg = edge(road_image_blur, 'canny')
    lines = hough_lines(road_image_blur_edg)
    transform = points_to_transform(img)
    line_image = draw_lines(transformed_image, lines)
    plt.subplot(2,1,1); imshow(transform)
    plt.subplot(2,1,2); imshow(line_image)
    plt.show()
    #imshow(m)
#%%
cap = imageio.get_reader('challenge.mp4')

i = 0
while(True):
    frame = cap.get_data(i)
    i += 1
    ld = line_detect(frame)
    plt.clf()
    #imshow(ld)
    print(i)
    #sleep(0.1)
    
    
    # Our operations on the frame come here
    #gray = rgb_to_(frame, 'gray')

    # Display the resulting frame
    #plt.imshow(gray)

# When everything done, release the capture
cap.release()
#%%
def morf(img):
    linek = np.zeros((3,3),dtype=np.uint8)
    linek[1,...]=1
    x=cv2.morphologyEx(img, cv2.MORPH_OPEN, linek ,iterations=1)
    return x
#kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(2,2))
#dst_edg = cv2.dilate(dst_edg,kernel,iterations = 2)
#imshow(dst_edg)
#kernel = np.ones((1,1),np.uint8)
#dst_edg = cv2.erode(dst_edg,kernel,iterations = 2)
#imshow(dst_edg)

#%%
def select_white_yellow(img):
    hsv = rgb_to_(img, 'hsv')
    # white color mask
    lower = np.uint8([  0, 200,   0])
    upper = np.uint8([255, 255, 255])
    white_mask = cv2.inRange(hsv, lower, upper)
    # yellow color mask
    lower = np.uint8([ 10,   0, 100])
    upper = np.uint8([ 40, 255, 255])
    yellow_mask = cv2.inRange(hsv, lower, upper)
    # combine the mask
    mask = cv2.bitwise_or(white_mask, yellow_mask)
    return cv2.bitwise_and(img, img, mask = mask)
#%%

#road_iamge = imread('solidWhiteRight.jpg')
imshow(frame)

transformed_image = four_point_prespective_transform(frame)
imshow(transformed_image)
#%%
road_image_gray = rgb_to_(transformed_image, 'gray')
#imshow(road_image_gray)

road_image_blur = gaussian_blur(road_image_gray)
#imshow(road_image_blur)

road_image_blur_edg = edge(road_image_blur, 'canny')
imshow(road_image_blur_edg)

lines = hough_lines(road_image_blur_edg)

transform = points_to_transform(frame)
#imshow(transform)

line_image = draw_lines(transformed_image, lines)
#imshow(line_image)
#%%