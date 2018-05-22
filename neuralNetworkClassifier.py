from sklearn.model_selection import train_test_split
from loadFaces import vectorize_data_faces,load_picture_captured
from saveReturnValuesCSV import return_of_image_and_rotule_vectors,return_subject_name
import numpy as np
from sklearn.preprocessing import normalize
from imageCapture import sample_capture_to_rank
import subprocess
from sklearn.neural_network import MLPClassifier

#inicializations
dataset_faces = return_of_image_and_rotule_vectors()
X, y = [dataset_faces[0],dataset_faces[1]]


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)

mlp = MLPClassifier()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
mlp.fit(X_train, y_train)

predictions = mlp.predict(X_test)

from sklearn.metrics import classification_report,confusion_matrix
#print(classification_report(y_test,predictions))
print(np.mean(y_test==predictions))