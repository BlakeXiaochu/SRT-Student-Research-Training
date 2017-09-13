from KCFtracker import *
import numpy as np
import cv2

testImg = cv2.imread('test_img.jpg')
testImg2 = cv2.imread('test_img_2.jpg')
newImg = cv2.resize(testImg, (60, 120))
newImg2 = cv2.resize(testImg2, (60, 120))

tracker = KCFtracker()
tracker.train(newImg, 1e-4, 'linear')

pos = tracker.detect(newImg2, 'linear')

print(pos)
