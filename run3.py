import cv2 as cv
import numpy as np
import glob


class_paths = glob.glob('dataset/training/*')
class_names = [name[17:] for name in class_paths]

print(cv.__version__)
img1 = cv.imread('dataset/testing/0.jpg')
gray = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
sift = cv.SIFT_create()
kp = sift.detect(gray, None)
img = cv.drawKeypoints(gray, kp, img1)
cv.imwrite('sift_keypoint.jpg', img)


