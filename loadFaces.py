import numpy as np
import cv2
from sklearn.model_selection import train_test_split

def load_pictureCaptured():
    filename = "frame"
    D = 100
    img_file = cv2.imread('frame.pgm', 0)
    img = cv2.resize(img_file, (D, D))
    img = np.reshape(img, (D * D))
    return img

def load_faces(Filename):
    # Number of Persons in DataBase
    N = 16

    # Number of image faces for each Person in Database
    Ni = 11

    # Data of images in vector format
    X = []

    # Target of each image (rotules)
    y = []

    # String of filename to concatenated
    filename_str = '{}'.format(Filename)
    str_1 = 'subject0'
    str_2 = 'subject'
    str_3 = ['.centerlight', '.glasses', '.happy', '.leftlight', '.noglasses',
             '.normal', '.rightlight', '.sad', '.sleepy', '.surprised', '.wink']
    str_4 = '.pgm'
    str_5 = '.jpg'

    # Dimensionality of Vector Image
    D = 100

    for i in range(N):  # Indice para os individuos

        for j in range(Ni):   # Indice para expressoes

            if i < 9:
                img_file = cv2.imread('{}/{}{}{}{}'.format(
                    filename_str, str_1, i + 1, str_3[j], str_4),0)
                # print('{} -> {}'.format((i+1), (j+1)))
                img = cv2.resize(img_file, (D, D))
                img = np.reshape(img, (D * D))
                X.append(img)
                y.append(i + 1)
            elif i >= 9 and i < 15:
                img_file = cv2.imread(
                    '{}/{}{}{}{}'.format(filename_str, str_2, i + 1, str_3[j], str_4), 0)
                img = cv2.resize(img_file, (D, D))
                img = np.reshape(img, (D * D))
                X.append(img)
                y.append(i + 1)
            else:
                img_file = cv2.imread('{}/{}{}{}{}'.format(
                    filename_str, str_2, i + 1, str_3[j], str_5), 0)
                img = cv2.resize(img_file, (D, D))
                img = np.reshape(img, (D * D))
                X.append(img)
                y.append(i + 1)


    return [X, y]
