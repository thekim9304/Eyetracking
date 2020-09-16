import cv2
import math
import numpy as np

import torch

from mobilenetv1_face_landmark import MobileNetV1_custom as mb_cus

def nothing(x):
    pass

def eye_on_mask(landmarks, mask, side):
    points = [landmarks[i] for i in side]
    points = np.array(points, dtype=np.int32)
    mask = cv2.fillConvexPoly(mask, points, 255)
    return mask

def contouring(thresh, mid, img, right=False):
    cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    try:
        cnt = max(cnts, key = cv2.contourArea) # finding contour with #maximum area
        M = cv2.moments(cnt)
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
        if right:
            cx += mid # Adding value of mid to x coordinate of centre of #right eye to adjust for dividing into two parts
        cv2.circle(img, (cx, cy), 2, (0, 0, 255), -1)# drawing over #eyeball with red
    except:
        pass

land_img_size = (224, 224, 3)
num_landmark = 136
model_landmark = mb_cus(land_img_size[0], land_img_size[-1], num_landmark).cuda().eval()
state_landmark = torch.load(f'Z:/pths/cropped_face_landmark_detection/test2/f_landmark-mobilev1_custom-{land_img_size}.pth',
                            map_location='cuda:0')
model_landmark.load_state_dict(state_landmark['model_state_dict'])

# img = cv2.imread('./HELEN_3057639344_1_0.jpg')
img = cv2.imread(f'E:/DB_FaceLandmark/300W-LP_only_face/imgs/HELEN_2990717111_1_1_flip.jpg')
x = img.astype(np.float32) / 255
x = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0).cuda()
landmark = model_landmark(x)
landmark = landmark[0]

img_draw = img.copy()
for ii in range(0, len(landmark), 2):
    img_draw = cv2.circle(img_draw,
                     (int(landmark[ii] * img.shape[1]),
                      int(landmark[ii + 1] * img.shape[0])),
                     2, (0, 0, 255), -1)

cv2.imshow('annotated', img_draw)
cv2.namedWindow('image')
cv2.createTrackbar('threshold', 'image', 0, 255, nothing)

left = [36, 37, 38, 39, 40, 41]
right = [42, 43, 44, 45, 46, 47]
landmarks = []
for i in range(0, len(landmark), 2):
    landmarks.append([landmark[i].item(), landmark[i+1].item()])
landmarks = (np.array(landmarks) * 224).astype(np.int)

while(True):
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    mask = eye_on_mask(landmarks, mask, left)
    mask = eye_on_mask(landmarks, mask, right)

    kernel = np.ones((9, 9), np.uint8)
    mask = cv2.dilate(mask, kernel, 5)
    eyes = cv2.bitwise_and(img, img, mask=mask)
    mask = (eyes == [0, 0, 0]).all(axis=2)
    eyes[mask] = [255, 255, 255]
    eyes_gray = cv2.cvtColor(eyes, cv2.COLOR_BGR2GRAY)

    threshold = cv2.getTrackbarPos('threshold', 'image')
    _, thresh = cv2.threshold(eyes_gray, threshold, 255, cv2.THRESH_BINARY)
    thresh = cv2.erode(thresh, None, iterations=2)
    thresh = cv2.dilate(thresh, None, iterations=4)
    thresh = cv2.medianBlur(thresh, 3)
    thresh = cv2.bitwise_not(thresh)

    mid = (landmarks[42][0] + landmarks[39][0]) // 2
    cv2.imshow('r', thresh[:, 0:mid])
    cv2.imshow('l', thresh[:, mid:])
    center_img = img.copy()
    contouring(thresh[:, 0:mid], mid, center_img)
    contouring(thresh[:, mid:], mid, center_img, True)

    cv2.imshow('image', thresh)
    cv2.imshow('eyes', center_img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()
