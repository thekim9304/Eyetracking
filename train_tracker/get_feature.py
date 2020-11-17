import csv
import numpy as np
# import pandas

data_path = 'C:/Users/th_k9/Desktop/Eyetracking/calibration/data'
subject_id = 'mjw'

f_100 = open(f'{data_path}/{subject_id}_100/{subject_id}_100_whole.csv', 'r', encoding='utf-8')
csv_100 = csv.reader(f_100)
info_100 = []
for i, line in enumerate(csv_100):
    if i == 2:
        break
    info_100 = line
f_100.close()

bbox_100 = list(map(int, info_100[2].split()))
l_center_100 = list(map(float, info_100[3].split()))
r_center_100 = list(map(float, info_100[4].split()))
landmarks_100 = list(map(int, info_100[5].split()))

print(f'bbox_100 : {bbox_100}')
print(f'l_center_100 : {l_center_100}')
print(f'r_center_100 : {r_center_100}')
print(f'landmarks_100 : {landmarks_100}\n')

f_150 = open(f'{data_path}/{subject_id}_150/{subject_id}_150_whole.csv', 'r', encoding='utf-8')
csv_150 = csv.reader(f_150)
info_150 = []
for i, line in enumerate(csv_150):
    if i == 2:
        break
    info_150 = line
f_150.close()

bbox_150 = list(map(int, info_150[2].split()))
l_center_150 = list(map(float, info_150[3].split()))
r_center_150 = list(map(float, info_150[4].split()))
landmarks_150 = list(map(int, info_150[5].split()))

print(f'bbox_150 : {bbox_150}')
print(f'l_center_150 : {l_center_150}')
print(f'r_center_150 : {r_center_150}')
print(f'landmarks_150 : {landmarks_150}')