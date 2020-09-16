import cv2
import time
import numpy as np
from PIL import Image

import sys

sys.path.append('C:/Users/th_k9/Desktop/Eyetracking/pytorch_facelandmark_detection')
from face_detection_model.mobilenetv1 import MobileNetV1
from face_detection_model.ssd import SSD, Predictor

import torch
import torch.nn as nn
from torchvision import models
import torchvision.transforms.functional as TF

left_eye = [36, 37, 38, 39, 40, 41]
right_eye = [42, 43, 44, 45, 46, 47]


points = [9, 16, 25]
init_x, init_y = 10, 10


class Network(nn.Module):
    def __init__(self, num_classes=136):
        super().__init__()
        self.model_name = 'resnet18'
        self.model = models.resnet18()
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        x = self.model(x)
        return x


def face_detection(frame, predictor):
    frame = cv2.resize(frame, (640, 480))
    prevTime = time.time()
    boxes, labels, probs = predictor.predict(frame, 1, 0.5)
    sec = time.time() - prevTime

    return boxes, labels, probs, sec


def landmark_detection(fl_detect_model, face, land_img_size):
    if land_img_size[-1] == 1 and face.shape[-1] == 3:
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    elif land_img_size[-1] == 3 and face.shape[-1] == 1:
        face = cv2.cvtColor(face, cv2.COLOR_GRAY2BGR)

    x = TF.resize(Image.fromarray(face), size=land_img_size[:2])
    x = TF.to_tensor(x)
    x = TF.normalize(x, [0.5], [0.5])

    prevTime = time.time()
    rel_landmark = fl_detect_model(x.unsqueeze(0).cuda())
    sec = time.time() - prevTime

    rel_landmark = np.array(rel_landmark.tolist()[0])

    return rel_landmark, sec


def draw_landmark(face, abs_landmarks):
    for i in range(0, abs_landmarks.shape[0], 2):
        face = cv2.circle(face, (int(abs_landmarks[i]), int(abs_landmarks[i + 1])), 2, (0, 0, 255), -1)

    return face


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
        cnt = max(cnts, key=cv2.contourArea)  # finding contour with #maximum area
        M = cv2.moments(cnt)
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        if right:
            cx += mid  # Adding value of mid to x coordinate of centre of #right eye to adjust for dividing into two parts
        cv2.circle(img, (cx, cy), 2, (0, 0, 255), -1)  # drawing over #eyeball with red
    except:
        pass


def shape_to_np(shape, dtype="int"):
    # initialize the list of (x, y)-coordinates
    coords = np.zeros((68, 2), dtype=dtype)
    # loop over the 68 facial landmarks and convert them
    # to a 2-tuple of (x, y)-coordinates
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    # return the list of (x, y)-coordinates
    return coords