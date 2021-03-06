import cv2
import dlib
import time
import numpy as np
from PIL import Image

import sys
sys.path.append('C:/Users/th_k9/Desktop/Eyetracking/pytorch_facelandmark_detection')
from face_detection_model.mobilenetv1 import MobileNetV1
from face_detection_model.ssd import SSD, Predictor
from mobilenetv1_face_landmark import MobileNetV1 as mb

import torch
import torch.nn as nn
from torchvision import models
import torchvision.transforms.functional as TF

from utils import *

click_pt = []

def click_event(event, x, y, flags, param):
    global click_pt

    if event == cv2.EVENT_LBUTTONDOWN:
        click_pt = [x, y]


def calib_board(n_point, i, j):
    whiteboard = np.ones((480, 640, 3))

    w_interval = (640 // (points[n_point] ** 0.5 - 1))
    h_interval = (480 // (points[n_point] ** 0.5 - 1))
    x = int(init_x + (w_interval * i) - 10)
    y = int(init_y + (h_interval * j) - 10)
    x = 10 if x < 10 else x
    x = 630 if x > 630 else x
    y = 10 if y < 10 else y
    y = 470 if y > 470 else y

    whiteboard = cv2.circle(whiteboard, (x, y), 7, (0, 0, 255), -1)

    cv2.namedWindow("whiteboard", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("whiteboard", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow("whiteboard", whiteboard)
    cv2.setMouseCallback("whiteboard", click_event)

    return x, y


def main():
    global click_pt

    f_detection_model = SSD(2, MobileNetV1(2), is_training=False)
    state = torch.load('Z:/pths/face_detection/ssd_mobilenetv1/ssd-mobilev1-face-2134_0.0192.pth')
    f_detection_model.load_state_dict(state['model_state_dict'])
    predictor = Predictor(f_detection_model, 300)

    fl_detect_model = Network()
    fl_detect_model = mb(in_size=224, in_ch=1, num_landmarks=136)
    fl_detect_model.load_state_dict(
        torch.load('Z:/pths/cropped_face_landmark_detection/IBUG_300W_large_face/test1/mobile1_(224, 1).pth',
                   map_location='cuda:0'))
    fl_detect_model.cuda().eval()

    land_predictor = dlib.shape_predictor('Z:/dlib/shape_68.dat')

    cv2.namedWindow('annotated')
    cv2.createTrackbar('threshold', 'annotated', 0, 255, nothing)
    cv2.createTrackbar('land_height', 'annotated', 0, 20, nothing)

    cap = cv2.VideoCapture(1)

    face_size = 300
    bbox_region = {'forehead': 35, 'chin': 0, 'add_face_width': 10}
    filters = {'bbox': 15, 'landmark': 3}

    face_detect, land_detect = False, False
    prev_landmark = []
    prev_x1, prev_x2, prev_y1, prev_y2 = 0, 0, 0, 0
    blackboard = np.full((480, face_size, 3), 0, dtype=np.uint8)
    n_point, i, j = 0, 0, 0
    while True:
        ret, frame = cap.read()

        if ret:
            boxes, labels, probs, sec1 = face_detection(frame, predictor)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            sec2 = 0.0

            '''Face & Landmark detection'''
            if boxes.size(0):
                box = boxes[0, :]
                label = f"Face: {probs[0]:.2f}"
                x1, x2, y1, y2 = int(box[0].item() - bbox_region['add_face_width']), int(box[2].item() + bbox_region['add_face_width']), int(
                    box[1].item() + bbox_region['forehead']), int(box[3].item() + bbox_region['chin'])
                x1 = 0 if x1 < 0 else x1

                if face_detect:
                    if abs(prev_x1 - x1) < filters['bbox']:
                        x1 = prev_x1
                    else:
                        prev_x1 = x1
                    if abs(prev_x2 - x2) < filters['bbox']:
                        x2 = prev_x2
                    else:
                        prev_x2 = x2
                    if abs(prev_y1 - y1) < filters['bbox']:
                        y1 = prev_y1
                    else:
                        prev_y1 = y1
                    if abs(prev_y2 - y2) < filters['bbox']:
                        y2 = prev_y2
                    else:
                        prev_y2 = y2
                else:
                    face_detect = True
                    prev_x1 = x1
                    prev_x2 = x2
                    prev_y1 = y1
                    prev_y2 = y2

                face = frame[y1:y2, x1:x2].copy()

                face_box = dlib.rectangle(left=x1, top=y1, right=x2, bottom=y2)
                prev_time = time.time()
                land_whole = land_predictor(gray, face_box)
                sec2 = time.time() - prev_time
                land_add = cv2.getTrackbarPos('land_height', 'annotated')
                land_whole = shape_to_np(land_whole, land_add=land_add)
                if land_detect:
                    idx = 0
                    for land, prev_land in zip(land_whole, prev_landmark):
                        if abs(land[0] - prev_land[0]) < filters['landmark']:
                            land_whole[idx][0] = prev_land[0]
                        else:
                            prev_landmark[idx][0] = land[0]
                        if abs(land[1] - prev_land[1]) < filters['landmark']:
                            land_whole[idx][1] = prev_land[1]
                        else:
                            prev_landmark[idx][1] = land[1]
                        idx += 1
                else:
                    land_detect = True
                    prev_landmark = land_whole

                for (x, y) in land_whole:
                    cv2.circle(frame, (x, y), 2, (0, 0, 255), -1)

                land_face = land_whole.copy()
                land_face[:, 0] = ((land_face[:, 0] - x1) / (x2 - x1)) * face_size
                land_face[:, 1] = ((land_face[:, 1] - y1) / (y2 - y1)) * face_size
                face = cv2.resize(face, (face_size, face_size))

                cv2.rectangle(frame, (x1, y1), (x2, y2), (128, 0, 255), 4)
                cv2.putText(frame, label, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            else:
                land_detect = False
                face_detect = False

            '''Detect eye center'''
            # 동공 중심 검출 코드 손보기
            if boxes.size(0):
                mask = np.zeros(face.shape[:2], dtype=np.uint8)
                mask = eye_on_mask(land_face, mask, left_eye)
                mask = eye_on_mask(land_face, mask, right_eye)
                kernel = np.ones((9, 9), np.uint8)
                mask = cv2.dilate(mask, kernel, 5)
                eyes = cv2.bitwise_and(face, face, mask=mask)
                mask = (eyes == [0, 0, 0]).all(axis=2)
                eyes[mask] = [255, 255, 255]
                eyes_gray = cv2.cvtColor(eyes, cv2.COLOR_BGR2GRAY)

                threshold = cv2.getTrackbarPos('threshold', 'annotated')
                _, thresh = cv2.threshold(eyes_gray, threshold, 255, cv2.THRESH_BINARY)
                thresh = cv2.erode(thresh, None, iterations=2)
                thresh = cv2.dilate(thresh, None, iterations=4)
                thresh = cv2.medianBlur(thresh, 3)
                thresh = cv2.bitwise_not(thresh)

                mid = (land_face[42][0] + land_face[39][0]) // 2
                l_center = contouring(thresh[:, 0:mid], mid, face)
                r_center = contouring(thresh[:, mid:], mid, face, True)

                cv2.imshow('image', thresh)
                blackboard[:300, :] = face
            ''''''

            '''whiteboard'''
            if j >= int(points[n_point] ** 0.5):
                j = 0
                n_point += 1
                if n_point > 2:
                    break
            if i >= int(points[n_point] ** 0.5):
                i = 0
                j += 1
            x, y = calib_board(n_point, i, j)
            if click_pt:
                if abs(click_pt[0] - x) < 2 and abs(click_pt[1] - y) < 2:
                    cal_pt = f'cal_{points[n_point]}_{j}_{i}'
                    print(cal_pt)
                    '''face landmark vector 저장'''
                    # landmark 제대로 검출됐는지도 추가
                    if boxes.size(0):
                        print(land_face)
                        print(f'l_center : {l_center}')
                        print(f'r_center : {r_center}')
                        print(f'x : {click_pt[0]}, y : {click_pt[1]}')
                        i += 1
                        click_pt.clear()
                    else:
                        click_pt.clear()
                    ''''''
                else:
                    click_pt.clear()
            ''''''

            cv2.putText(frame, f'face detection : {sec1 * 100:.2}ms', (10, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 0, 255), 2)
            cv2.putText(frame, f'landmark detection : {sec2 * 100:.2}ms', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 0, 255), 2)
            concat_img = cv2.hconcat([blackboard, frame])
            cv2.imshow('annotated', concat_img)

            if cv2.waitKey(1) == 27:
                break
        else:
            break

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()