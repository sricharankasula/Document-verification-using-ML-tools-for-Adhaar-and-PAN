#!/usr/bin/env python
# coding: utf-8

import re
import cv2 
import numpy as np
import pytesseract
from pytesseract import Output
class Preprocess():
	def __init__(self, image):
	   self.image = image
	   
	# get grayscale image
	def get_grayscale(self):
	    return cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

	# noise removal
	def remove_noise(self):
	    return cv2.medianBlur(self.image,1)

	#adptive threshold
	def adathresh(self):
	    return cv2.adaptiveThreshold(self.image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 115, 1)
	 
	#thresholding
	def thresholding(self):
	    return cv2.threshold(self.image, 0, 200, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]


	#dilation
	def dilate(self):
	    kernel = np.ones((5,5),np.uint8)
	    return cv2.dilate(self.image, kernel, iterations = 1)
	    
	#erosion
	def erode(self):
	    kernel = np.ones((5,5),np.uint8)
	    return cv2.erode(self.image, kernel, iterations = 1)

	#opening - erosion followed by dilation
	def opening(self):
	    kernel = np.ones((5,5),np.uint8)
	    return cv2.morphologyEx(self.image, cv2.MORPH_OPEN, kernel)

	#canny edge detection
	def canny(self):
	    return cv2.Canny(self.image, 100, 200)

	# #skew correction
	# def deskew(self,image):
	#     coords = np.column_stack(np.where(self.image > 0))
	#     angle = cv2.minAreaRect(coords)[-1]
	#     if angle < -45:
	# 	   angle = -(90 + angle)
	#     else:
	# 	    angle = -angle
	# 	    (h, w) = image.shape[:2]
	# 	    center = (w // 2, h // 2)
	# 	    M = cv2.getRotationMatrix2D(center, angle, 1.0)
	# 	    rotated = cv2.warpAffine(self.image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
	#     return rotated

	# #template matching
	# def match_template(self,image, template):
	#     return cv2.matchTemplate(self.image, template, cv2.TM_CCOEFF_NORMED) 


