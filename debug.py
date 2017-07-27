# -*- coding: utf-8 -*-
"""
Created on Sun Jul 23 12:09:42 2017

@author: ASUS
"""

from image_processing import *
from helper import *

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

from scipy import interpolate

x = np.linspace(0,10,30)
y = np.exp(-x/4)*x

f = interpolate.interp1d(x,y)
xnew = np.linspace(0,10,100)

plt.clf()
plt.plot(x,y, 'r*', xnew, f(xnew), 'b-')