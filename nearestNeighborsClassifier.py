from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn.model_selection import train_test_split
from loadFaces import load_faces,load_picture_captured
import numpy as np
from sklearn.preprocessing import normalize
#from imageCapture import sample_capture

#inicializations
dataset_faces = load_faces('centered_faces')
X, y = [dataset_faces[0], dataset_faces[1]]

X = normalize(X)

neighKNeigh = KNeighborsClassifier(n_neighbors=3)

def nearest_knearest_neighborhood_training():
	
	Nr = 0
	hitsVector = []

	while Nr != 100:
		print("Rodada {}".format(Nr+1))
		X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
		neighKNeigh.fit(X_train, y_train)

		count = 0
		predict = neighKNeigh.predict(X_test)
		hits = 0

		for index in y_test:
			#print('Actual: {} -> Predict {}'.format(index, predict[count]))

			if(index == predict[count]):
				hits = hits + 1
			count = count + 1

		hitsVector.append((hits / len(y_test)) * 100)
		Nr += 1

	print("Mediana de acertos :{} %".format(np.median(np.array(hitsVector))))
	print("Media de acertos :{} %".format(np.mean(np.array(hitsVector))))
	print("Minimo acerto :{} %".format(np.min(np.array(hitsVector))))
	print("Maximo acerto :{} %".format(np.max(np.array(hitsVector))))


#nearest_knearest_neighborhood_training() 

#take a picture for classification
sample_capture()

# read the image that had captured
img = load_picture_captured()
img = img.reshape(1,-1) # for convert in a single sample.


# Normalizar imagem capturada e o dataset
img = normalize(img)



neigh.fit(X,y)
predict = neigh.predict(img)
print("The system predicts :{}".format(predict))
