import cv2
import numpy as np
cascade = cv2.CascadeClassifier('resources/haarcascade_frontalface_default.xml')
img = cv2.imread('resources/birthday.jpg')
img = cv2.resize(img, (800, 500))
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

faces = cascade.detectMultiScale(image=gray_img, scaleFactor=1.1, minNeighbors=6)

for (x, y, width, height) in faces:
    cv2.rectangle(img=img, pt1=(x, y), pt2=(x+width, y+height), color=(0, 0, 255), thickness=3)

cv2.imshow('Photo', img)
cv2.waitKey(0)
cv2.destroyAllWindows()