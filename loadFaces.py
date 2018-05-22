import numpy as np
import cv2
from sklearn.model_selection import train_test_split
import pandas as pd
from detectFaces import detect_faces
import imutils
import dlib
from imutils import face_utils

# Inicializations
D = 300 # Dimension to reshape images, when vectorize one vector -> len = 90000 (300x300)

def load_picture_captured():
    img_file = cv2.imread('sample_to_rank.png', 0)
    img = cv2.resize(img_file, (D, D))
    img = np.reshape(img, (D * D))
    return img

def vectorize_data_faces_cutting(Filename,numberOfPersons):
    # Number of Persons in DataBase
    N = numberOfPersons 

    # Number of image faces for each Person in Database
    Ni = 10
        # String of filename to concatenated
    filename_str = '{}'.format(Filename)
    str_1 = ['Subject_','subject0','subject']
    str_3 = ['.pgm','.png','.jpg']
    str_4 = '_type_'

    haar_face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')

    #call our function to detect faces 
    
    for i in range(N):  # Indice para os individuos
        
        for j in range(Ni):   # Indice para expressoes

            if i < 9:
                str_ = '{}/{}{}{}{}{}'.format(filename_str, str_1[0], i + 1, str_4, j + 1, str_3[1])
                img_file = cv2.imread(str_,0)
                faces_detected_img = detect_faces(haar_face_cascade, img_file)  
                cv2.imwrite(str_,faces_detected_img)
                #print(str_)


def vectorize_data_faces(Filename,numberOfPersons):
    # Number of Persons in DataBase
    N = numberOfPersons 

    # Number of image faces for each Person in Database
    Ni = 10

    # Data of images in vector format
    X = []

    # Target of each image (rotules)
    y = []

    # String of filename to concatenated
    filename_str = '{}'.format(Filename)
    str_1 = ['Subject_','subject0','subject']
    str_3 = ['.pgm','.png','.jpg']
    str_4 = '_type_'

    for i in range(N):  # Indice para os individuos
        
        for j in range(Ni):   # Indice para expressoes

            if j < 10:
                img_file = cv2.imread('{}/{}{}{}{}{}'.format(filename_str, str_1[0], i + 1, str_4, j + 1, str_3[1]),0)
                #print('{}'.format(img_file))
                #print(np.size(img_file))
                img = cv2.resize(img_file, (D, D))
                img = np.reshape(img, (D * D))
                X.append(img)
                y.append(i + 1)
            
    return [X, y]

def vectorize_data_faces_cutting_aligned(Filename,numberOfPersons):
    # Number of Persons in DataBase
    N = numberOfPersons 

    # Number of image faces for each Person in Database
    Ni = 10

    # Data of images in vector format
    X = []

    # Target of each image (rotules)
    y = []

    # String of filename to concatenated
    filename_str = '{}'.format(Filename)
    str_1 = ['Subject_','subject0','subject']
    str_3 = ['.pgm','.png','.jpg']
    str_4 = '_type_'

    for i in range(N):  # Indice para os individuos
        
        for j in range(Ni):   # Indice para expressoes

            if j < 10:
                img_file = cv2.imread('{}/{}{}{}{}{}'.format(filename_str, str_1[0], i + 1, str_4, j + 1, str_3[1]),0)
                #print('{}'.format(img_file))
                #print(np.size(img_file))
                img = cv2.resize(img_file, (D, D))
                img = np.reshape(img, (D * D))
                X.append(img)
                y.append(i + 1)
            
    return [X, y]