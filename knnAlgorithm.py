from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from loadFaces import load_faces,load_pictureCaptured
import numpy as np
from sklearn.preprocessing import normalize
from videoCapture import video_capture

#take a picture for classification
#video_capture()

# read the image that had captured
img = load_pictureCaptured()
img = img.reshape(1,-1) # for convert in a single sample.


# Load Data Face Sets
dataset_faces = load_faces('centered_faces [test]')
X, y = [dataset_faces[0], dataset_faces[1]]

# Normalizar capturedImage and dataset
img = normalize(img)
X = normalize(X)

Nr = 100

neigh = KNeighborsClassifier(n_neighbors=3)
#neigh.fit(X_train, y_train)
#predict = neigh.predict(X_test)

neigh.fit(X,y)
predict = neigh.predict(img)
print("The system predicts :{}".format(predict))
