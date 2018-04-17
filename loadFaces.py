import numpy as np
import cv2
from sklearn.model_selection import train_test_split

def load_picture_captured():
    D = 100
    img_file = cv2.imread('sample_to_rank.png', 0)
    img = cv2.resize(img_file, (D, D))
    img = np.reshape(img, (D * D))
    return img

def load_faces(Filename):
    # Number of Persons in DataBase
    N = 4

    # Number of image faces for each Person in Database
    Ni = 10

    # Data of images in vector format
    X = []

    # Target of each image (rotules)
    y = []

    # String of filename to concatenated
    filename_str = '{}'.format(Filename)
    str_1 = ['Subject0','subject0','subject']
    str_2 = ['.centerlight', '.glasses', '.happy', '.leftlight', '.noglasses',
             '.normal', '.rightlight', '.sad', '.sleepy', '.surprised', '.wink']
    str_3 = ['.pgm','.png','.jpg']
    str_4 = '-type '

    # Dimensionality of Vector Image
    D = 100

    for i in range(N):  # Indice para os individuos
        
        for j in range(Ni):   # Indice para expressoes

            if i < 9:
                str_ = '{}/{}{}{}{}{}'.format(filename_str, str_1[0], i + 1, str_4, j+1, str_3[1])
                #print(str_)
                img_file = cv2.imread(str_,0)
                img = cv2.resize(img_file, (D, D))
                img = np.reshape(img, (D * D))
                X.append(img)
                y.append(i + 1)
            #elif i >= 9 and i < 15:
            #    img_file = cv2.imread(
            #        '{}/{}{}{}{}'.format(filename_str, str_2, i + 1, str_3[j], str_4), 0)
            #    img = cv2.resize(img_file, (D, D))
            #    img = np.reshape(img, (D * D))
            #    X.append(img)
            #    y.append(i + 1)
            #else:
            #    img_file = cv2.imread('{}/{}{}{}{}'.format(
            #        filename_str, str_2, i + 1, str_3[j], str_5), 0)
            #    img = cv2.resize(img_file, (D, D))
            #    img = np.reshape(img, (D * D))
            #    X.append(img)
            #    y.append(i + 1)

    
    return [X, y]
