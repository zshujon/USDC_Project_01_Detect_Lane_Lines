# -*- coding: utf-8 -*-
"""
Created on Mon Jul 24 18:47:34 2017

@author: ZSHUJON
"""

from image_processing import np, cv2

#%%
#==============================================================================
# TRANSFORM_POINTS_00 = [100,500]
# TRANSFORM_POINTS_01 = [360,350]
# TRANSFORM_POINTS_02 = [600,350]
# TRANSFORM_POINTS_03 = [920,500]
#==============================================================================

TRANSFORM_POINTS_00 = [285,630]
TRANSFORM_POINTS_01 = [550,500]
TRANSFORM_POINTS_02 = [750,500]
TRANSFORM_POINTS_03 = [1075,630]

def points_to_transform(img):
    pts = np.array([TRANSFORM_POINTS_00, TRANSFORM_POINTS_01, TRANSFORM_POINTS_02, TRANSFORM_POINTS_03], np.int32)
    ptss = pts.reshape((-1,1,2))
    transformed = cv2.polylines(img.copy(),[ptss],True,(0,255,255), 2)
    return transformed

def draw_lines(img, lines):    
    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(img,(x1,y1),(x2,y2),(0,255,140),5)
    return img