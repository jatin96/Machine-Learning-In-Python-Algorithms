import cv2
import numpy as np
img = cv2.imread('bookpage.jpg')
grayscaled = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
retval, threshold = cv2.threshold(grayscaled, 12, 255, cv2.THRESH_BINARY)
gauss = cv2.adaptiveThreshold(grayscaled, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 57, 2)
cv2.imshow('original image', img)
cv2.imshow('threshold image', threshold)
cv2.imshow('gaussian threshold', gauss)
cv2.waitKey(0)
cv2.destroyAllWindows()
