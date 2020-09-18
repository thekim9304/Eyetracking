import os
import cv2
import numpy as np

root_path = '../calibration/eye_img'
img_paths = os.listdir(root_path)

img_path = img_paths[0]

img = cv2.imread(f'{root_path}/{img_path}', 0)
img = cv2.resize(img, (100, 100))
img2 = cv2.equalizeHist(img)

# Canny edge -> Sobel mask
edge = cv2.Canny(img, 199, 210)

# circles3 = cv2.HoughCircles(edge, cv2.HOUGH_GRADIENT, 1, 20, param1=50, param2=30, minRadius=0, maxRadius=0)
# circles3 = np.uint16(np.around(circles3))
# cimg3 = cv2.cvtColor(edge, cv2.COLOR_GRAY2BGR)
# for i in circles3[0,:]:
#     cv2.circle(cimg3, (i[0], i[1]), i[2], (0, 255, 0), 2)

concat_img_1 = cv2.vconcat([img, img2, edge])
# concat_img_3 = cv2.vconcat([cimg1, cimg2])

cv2.imshow('ori', img)
cv2.imshow('equalHist', img2)
cv2.imshow('result', concat_img_1)
# cv2.imshow('result2', concat_img_3)
cv2.waitKey()
cv2.destroyAllWindows()