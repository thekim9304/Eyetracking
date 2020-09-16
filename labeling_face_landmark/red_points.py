import cv2
import numpy as np

img = cv2.imread('1.jpg')
# img[180:190, 210:230] = [0, 0, 255]
# img[200:210, 210:230] = [15, 17, 212]
cv2.imshow('t', img)
cv2.waitKey()
cv2.destroyAllWindows()



for j in range(200, 210):
    for i in range(210, 230):
        print(img[j][i])

