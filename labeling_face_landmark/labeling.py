import os
import cv2
import csv

def draw_circle(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        landmarks.append([x, y])
        print(landmarks)

    if event == cv2.EVENT_RBUTTONDOWN:
        if landmarks:
            landmarks.pop()
            print(landmarks)

labels = []

img_path = './img_th'
imgs = os.listdir(img_path)
for img_id in imgs:
    img = cv2.imread(f'./{img_path}/{img_id}')
    img = cv2.resize(img, (450, 450))

    cv2.namedWindow('image')
    cv2.setMouseCallback('image', draw_circle)

    landmarks = []
    while True:
        draw_img = img.copy()
        if landmarks:
            for i, landmark in enumerate(landmarks):
                cv2.circle(draw_img, (landmark[0], landmark[1]), 2, (0, 0, 255), -1)
                cv2.putText(draw_img, f'{i}', (landmark[0], landmark[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)

        cv2.imshow('image', draw_img)

        k = cv2.waitKey(1) & 0xFF
        if k == 113:
            cv2.imwrite(f'./label_check/{img_id}', draw_img)

            str_land = ''
            for landmark in landmarks:
                for land in landmark:
                    str_land += str(land)
                    str_land += ' '

            labels.append([img_id, str_land[:-1]])

            f = open('jw_landmark.csv', 'w', encoding='utf-8', newline='')
            writer = csv.writer(f)
            writer.writerow(['ImageID', 'width', 'height' 'Landmarks'])
            for label in labels:
                writer.writerow([label[0], '450', '450', label[1]])

            f.close()

            break

    cv2.destroyAllWindows()