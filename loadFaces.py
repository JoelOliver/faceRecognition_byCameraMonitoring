import numpy as np
import cv2
from sklearn.model_selection import train_test_split
import pandas as pd

def load_picture_captured():
    D = 100
    img_file = cv2.imread('sample_to_rank.png', 0)
    img = cv2.resize(img_file, (D, D))
    img = np.reshape(img, (D * D))
    return img

def vectorize_data_faces(Filename,last_index_subject):
    # Number of Persons in DataBase
    N = last_index_subject + 1

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

    # Dimensionality of Vector Image
    D = 100

    for i in range(N):  # Indice para os individuos
        
        for j in range(Ni):   # Indice para expressoes

            if i < 9:
                img_file = cv2.imread('{}/{}{}{}{}'.format(
                    filename_str, str_1, i + 1, str_3[j], str_4),0)
                # print('{} -> {}'.format((i+1), (j+1)))

                print(np.size(img_file))
                img = cv2.resize(img_file, (D, D))
                img = np.reshape(img, (D * D))
                X.append(img)
                y.append(i + 1)
            elif i >= 9 and i < 15:
                img_file = cv2.imread(
                    '{}/{}{}{}{}'.format(filename_str, str_2, i + 1, str_3[j], str_4), 0)
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

def save_vectorized_load_faces_in_csv_file(v):
    #dataset_faces = vectorize_data_faces('samples_faces_dataset')
    #X, y = [dataset_faces[0], dataset_faces[1]]
    X, y = [v[0],v[1]]
    X_df = pd.DataFrame(X)
    X_df.to_csv('X.csv', index=False, header=False)

    y_df = pd.DataFrame(y)  
    y_df.to_csv('y.csv', index=False, header=False)

    print("Imagens vetorizadas e salvas em arquivos do tipo .csv")

#save_load_faces_in_csv_file()
#df = pd.read_csv('X.csv')
#print(len(df.values.tolist()[0]))

def return_of_image_vectors():
    X = pd.read_csv('X.csv').values.tolist()
    y = pd.read_csv('y.csv').values.tolist()

    return [X,y]

def return_last_index_subject():
    last_index_subject = pd.read_csv('y.csv').values.tolist()[-1][0]

    return last_index_subject

#print(return_last_index_subject())    
