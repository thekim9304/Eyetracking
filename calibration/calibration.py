import os
import csv
import sys
from tkinter import messagebox

from utils import *

click_pt = []


def click_event(event, x, y, flags, param):
    global click_pt

    if event == cv2.EVENT_LBUTTONDOWN:
        click_pt = [x, y]


def calib_board(n_point, i, j):
    whiteboard = np.full((480, 640), 255, np.float32)
    whiteboard = cv2.cvtColor(whiteboard, cv2.COLOR_GRAY2BGR)

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
    cv2.setWindowProperty(
        "whiteboard", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow("whiteboard", whiteboard)
    cv2.setMouseCallback("whiteboard", click_event)

    return x, y


def main():
    global click_pt

    # setting
    # 70cm, 100cm, 150cm
    subject = '11'
    num_save_frame = 10

    root_path = f'./data/{subject}'
    face_img_path = f'{root_path}/face_img'
    whole_img_path = f'{root_path}/whole_img'
    check_img_path = f'{root_path}/check_img'

    if not os.path.exists(root_path):
        os.mkdir(root_path)
        if not os.path.exists(face_img_path):
            os.mkdir(face_img_path)
        if not os.path.exists(whole_img_path):
            os.mkdir(whole_img_path)
        if not os.path.exists(check_img_path):
            os.mkdir(check_img_path)
    else:
        pass
        # messagebox.showinfo("중복", "이미 존재하는 subject")
        # sys.exit()

    face_csv_name = f'{subject}_face'
    f_face = open(f'./data/{subject}/{face_csv_name}.csv',
                  'w', encoding='utf-8', newline='')
    wr_face = csv.writer(f_face)
    wr_face.writerow(['id', 'img_size(h,w)', 'l_eye_center',
                      'r_eye_center', 'nose_landmarks'])
    whole_csv_name = f'{subject}_whole'
    f_whole = open(f'./data/{subject}/{whole_csv_name}.csv',
                   'w', encoding='utf-8', newline='')
    wr_whole = csv.writer(f_whole)
    wr_whole.writerow(['id', 'img_size(h,w)', 'bbox(x1,y1,x2,y2)',
                       'l_eye_center', 'r_eye_center', 'total_landmarks'])

    # face detector
    pth_path = 'F:/pths/face_detection/ssd_mobilenetv1/ssd-mobilev1-face-2134_0.0192.pth'
    face_detector = face_detector_loader(pth_path)
    # landmark detector
    dat_path = 'F:/dlib/shape_68.dat'
    land_detector = dlib.shape_predictor(dat_path)

    cap = cv2.VideoCapture(0)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    cv2.namedWindow('annotated')
    cv2.createTrackbar('threshold', 'annotated', 0, 255, nothing)
    cv2.createTrackbar('land_height', 'annotated', 0, 20, nothing)

    # init value
    clicked = False
    frame_cnt = 0
    face_detect, land_detect = False, False
    prev_bbox, prev_land = [], []
    n_point, i, j = 0, 0, 0
    while True:
        ret, frame = cap.read()
        ori_frame = frame.copy()

        if ret:
            blackboard = np.full(
                (face_size * 2, (face_size + frame_width), 3), 0, dtype=np.uint8)
            boxes, labels, probs, sec1 = get_face(face_detector, frame)
            sec2 = 0.0
            abs_land, l_center, r_center = [], [], []

            if boxes.size(0) and probs[0] > 0.5:
                box = boxes[0, :]
                label = f"Face: {probs[0]:.2f}"

                cur_bbox = add_face_region(box)
                cur_bbox, prev_bbox, face_detect = low_pass_filter(
                    cur_bbox, prev_bbox, face_detect, mode='face')
                x1, x2, y1, y2 = cur_bbox

                '''detect landmark'''
                cur_land, sec2 = get_landmark(land_detector, frame, cur_bbox)
                land_add = cv2.getTrackbarPos('land_height', 'annotated')
                cur_land = cvt_shape_to_np(cur_land, land_add=land_add)
                cur_land, prev_land, land_detect = low_pass_filter(
                    cur_land, prev_land, land_detect, mode='landmark')
                cur_rel_coord = cvt_land_rel(cur_land, cur_bbox)

                '''detect eyeball'''
                ori_face = frame[y1:y2, x1:x2].copy()
                face = frame[y1:y2, x1:x2].copy()
                face = cv2.resize(face, (face_size, face_size))
                abs_land = (cur_rel_coord * face_size).astype(np.int)
                centers, thresh = get_eye_centers(face, cur_rel_coord)
                if 0 not in centers[0] and 0 not in centers[1]:
                    for land_i in range(27, 36):
                        face = cv2.line(
                            face, centers[0], (abs_land[land_i][0], abs_land[land_i][1]), (255, 0, 0), 1)
                        face = cv2.line(
                            face, centers[1], (abs_land[land_i][0], abs_land[land_i][1]), (255, 255, 0), 1)
                    l_center = np.array(centers[0]) / face.shape[0]
                    r_center = np.array(centers[1]) / face.shape[0]

                    # draw on original frame
                    w_l_center = l_center.copy()
                    w_r_center = r_center.copy()
                    w_l_center[0] = w_l_center[0] * (x2 - x1) + x1
                    w_l_center[1] = w_l_center[1] * (y2 - y1) + y1
                    w_r_center[0] = w_r_center[0] * (x2 - x1) + x1
                    w_r_center[1] = w_r_center[1] * (y2 - y1) + y1
                    frame = cv2.circle(frame, (int(w_l_center[0]), int(w_l_center[1])), 2,
                                       (255, 255, 255), -1)
                    frame = cv2.circle(frame, (int(w_r_center[0]), int(w_r_center[1])), 2,
                                       (255, 255, 255), -1)

                # draw on blackboard
                cv2.rectangle(frame, (x1, y1), (x2, y2), (128, 0, 255), 4)
                cv2.putText(frame, label, (x1, y1),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                frame = draw_land(frame, cur_land, (0, 0, 255))
                blackboard[:face_size, :face_size] = face
                blackboard[face_size:, :face_size] = thresh
            else:
                land_detect = False
                face_detect = False

            '''data save'''
            if clicked and frame_cnt < num_save_frame:
                if frame_cnt == 0:
                    id = f'{points[n_point]}_{i+ int(j * (points[n_point] ** 0.5))}'
                    print(f'{id} was clicked.')

                cv2.imwrite(
                    f'{whole_img_path}/whole_{id}_{frame_cnt}.jpg', ori_frame)
                cv2.imwrite(
                    f'{face_img_path}/face_{id}_{frame_cnt}.jpg', ori_face)
                cv2.imwrite(
                    f'{check_img_path}/check_{id}_{frame_cnt}.jpg', face)

                # wr_face.writerow(['id', 'img_size', 'l_eye_center', 'r_eye_center', 'nose_landmarks'])
                # wr_whole.writerow(['id', 'img_size', 'l_eye_center', 'r_eye_center', 'total_landmarks'])
                f_size = list_to_str(face.shape[:2])
                f_l_center = list_to_str(l_center)  # relative
                f_r_center = list_to_str(r_center)  # relative
                f_rel_coord = np_to_str(cur_rel_coord[idx_nose])  # relative
                wr_face.writerow(
                    [f'face_{id}_{frame_cnt}.jpg', f_size, f_l_center, f_r_center, f_rel_coord])
                w_size = list_to_str(ori_frame.shape[:2])
                w_bbox = list_to_str((x1, y1, x2, y2))  # absolute
                w_l_center = list_to_str(w_l_center)  # absolute
                w_r_center = list_to_str(w_r_center)  # absolute
                w_cur_coord = np_to_str(cur_land)  # absolute
                wr_whole.writerow(
                    [f'whole_{id}_{frame_cnt}.jpg', w_size, w_bbox, w_l_center, w_r_center, w_cur_coord])

                frame_cnt += 1
            else:
                clicked = False
                frame_cnt = 0

            '''whiteboard'''
            if not clicked:
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
                    if abs(click_pt[0] - x) < 5 and abs(click_pt[1] - y) < 5:
                        '''face landmark vector 저장'''
                        if land_detect and len(l_center) and len(r_center):
                            clicked = True
                            frame_cnt = 0

                            i += 1
                            click_pt.clear()
                        else:
                            print('Not detected features !')
                            click_pt.clear()
                        ''''''
                    else:
                        click_pt.clear()
            ''''''

            '''draw result'''
            notice_board = draw_speed(
                (frame_height, frame_width), (sec1, sec2))
            blackboard[frame_height:, face_size:] = notice_board
            blackboard[:frame_height, face_size:] = frame
            cv2.imshow('annotated', blackboard)

            key = cv2.waitKey(1)
            if key == 27:
                break
        else:
            break

    cv2.destroyAllWindows()
    cap.release()
    f_face.close()
    f_whole.close()


if __name__ == '__main__':
    main()
