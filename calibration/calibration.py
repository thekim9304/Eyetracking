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

    cap = cv2.VideoCapture(0)

    bbox_region = {'forehead': 0, 'chin': 0, 'add_face_width': 10}
    filters = {'bbox': 10, 'landmark': 5}

    face_detect, land_detect = False, False
    prev_landmark = []
    rel_landmark = []
    prev_x1, prev_x2, prev_y1, prev_y2 = 0, 0, 0, 0
    blackboard = np.full((480, 300, 3), 0, dtype=np.uint8)
    n_point, i, j = 0, 0, 0
    while True:
        ret, frame = cap.read()

        if ret:
            boxes, labels, probs, sec1 = face_detection(frame, predictor)
            draw_img = frame.copy()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            sec2 = 0.0

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
                face2 = frame[y1:y2, x1:x2].copy()

                rel_landmark, sec2 = landmark_detection(fl_detect_model, face, (224, 224, 1))

                # face_box = dlib.rectangle(left=x1, top=y1, right=x2, bottom=y2)
                # shape = land_predictor(gray, face_box)
                # shape = shape_to_np(shape)
                # for (x, y) in shape:
                #     cv2.circle(draw_img, (x, y), 2, (0, 0, 255), -1)

                draw_face_size = 300
                face = cv2.resize(face, (draw_face_size, draw_face_size))
                abs_landmark = rel_landmark.copy()
                abs_landmark[0::2] = (abs_landmark[0::2] + 0.5) * draw_face_size
                abs_landmark[1::2] = (abs_landmark[1::2] + 0.5) * draw_face_size

                if land_detect:
                    idx = 0
                    for land, prev_land in zip(abs_landmark, prev_landmark):
                        if abs(land - prev_land) < filters['landmark']:
                            abs_landmark[idx] = prev_landmark[idx]
                        else:
                            prev_landmark[idx] = abs_landmark[idx]
                        idx += 1
                else:
                    land_detect = True
                    prev_landmark = abs_landmark

                if face.shape[-1] != 3:
                    face = cv2.cvtColor(face, cv2.COLOR_GRAY2BGR)
                face = draw_landmark(face, abs_landmark)
                blackboard[:300, :] = face

                cv2.rectangle(frame, (x1, y1), (x2, y2), (128, 0, 255), 4)
                cv2.putText(frame, label, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            else:
                land_detect = False
                face_detect = False

            cv2.putText(frame, f'face detection : {sec1 * 100:.2}ms', (10, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 0, 255), 2)
            cv2.putText(frame, f'landmark detection : {sec2 * 100:.2}ms', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 0, 255), 2)
            concat_img = cv2.hconcat([blackboard, frame])
            cv2.imshow('annotated', concat_img)

            # cv2.imshow('draw_img', draw_img)

            '''Detect eye center'''
            # rel_landmark
            if boxes.size(0):
                size = 300
                face2 = cv2.resize(face2, (size, size))
                land_croped_face = []
                for r_land in range(0, len(rel_landmark), 2):
                    land_croped_face.append([int((rel_landmark[r_land] + 0.5) * size),
                                             int((rel_landmark[r_land+1] + 0.5) * size)])
                mask = np.zeros(face2.shape[:2], dtype=np.uint8)
                mask = eye_on_mask(land_croped_face, mask, left_eye)
                mask = eye_on_mask(land_croped_face, mask, right_eye)
                kernel = np.ones((9, 9), np.uint8)
                mask = cv2.dilate(mask, kernel, 5)
                eyes = cv2.bitwise_and(face2, face2, mask=mask)
                mask = (eyes == [0, 0, 0]).all(axis=2)
                eyes[mask] = [255, 255, 255]
                eyes_gray = cv2.cvtColor(eyes, cv2.COLOR_BGR2GRAY)

                threshold = cv2.getTrackbarPos('threshold', 'annotated')
                _, thresh = cv2.threshold(eyes_gray, threshold, 255, cv2.THRESH_BINARY)
                thresh = cv2.erode(thresh, None, iterations=2)
                thresh = cv2.dilate(thresh, None, iterations=4)
                thresh = cv2.medianBlur(thresh, 3)
                thresh = cv2.bitwise_not(thresh)

                mid = (land_croped_face[42][0] + land_croped_face[39][0]) // 2
                center_img = face2.copy()
                contouring(thresh[:, 0:mid], mid, center_img)
                contouring(thresh[:, mid:], mid, center_img, True)

                cv2.imshow('image', thresh)
                cv2.imshow('eyes', center_img)
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
                        print(rel_landmark)
                        print(f'x : {click_pt[0]}, y : {click_pt[1]}')
                        i += 1
                        click_pt.clear()
                    else:
                        click_pt.clear()
                    ''''''
                else:
                    click_pt.clear()
            ''''''

            if cv2.waitKey(1) == 27:
                break
        else:
            break

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()