import pandas as pd
import numpy as np
import subprocess
import cv2


X = []

img_file = cv2.imread('samples_faces_dataset/Subject_1_type_1.png',0)
print(img_file.shape)
#print('{}'.format(img_file))
#print(np.size(img_file))
#img = cv2.resize(img_file, (D, D))
#img = np.reshape(img_file, (img_file.shape[0] * img_file.shape[1]))
#print(len(img))
#X.append(img)

img_file = cv2.imread('samples_faces_dataset/Subject_3_type_1.png',0)
print(img_file.shape)

