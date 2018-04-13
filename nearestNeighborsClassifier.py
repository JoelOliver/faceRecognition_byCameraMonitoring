from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from loadFaces import load_faces,load_picture_captured
import numpy as np
from sklearn.preprocessing import normalize
from imageCapture import image_capture

#take a picture for classification
image_capture()

# read the image that had captured
img = load_picture_captured()
img = img.reshape(1,-1) # for convert in a single sample.

# Load Data Face Sets
dataset_faces = load_faces('centered_faces [test]')
X, y = [dataset_faces[0], dataset_faces[1]]

# Normalizar imagem capturada e o dataset
img = normalize(img)
X = normalize(X)

neigh = KNeighborsClassifier(n_neighbors=3)

neigh.fit(X,y)
predict = neigh.predict(img)
print("The system predicts :{}".format(predict))
