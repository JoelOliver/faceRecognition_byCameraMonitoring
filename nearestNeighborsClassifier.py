from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn.model_selection import train_test_split
from loadFaces import load_faces,load_picture_captured
import numpy as np
from sklearn.preprocessing import normalize
from imageCapture import sample_capture_to_rank

#inicializations
dataset_faces = load_faces('samples_faces_dataset')
X, y = [dataset_faces[0], dataset_faces[1]]

X = normalize(X)

# para mudar valores dos parâmetros, verificar documentação do scikit-learn
neighKNeigh = KNeighborsClassifier(n_neighbors=3)
neighCentroid = NearestCentroid()

def knearest_neighborhood_training():
	print("Rodadas de treinamento - Classificador KNearest Neighborhood")
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

#Testar função -> nearest_knearest_neighborhood_training() 
#knearest_neighborhood_training() 

def centroid_training():
	print("Rodadas de treinamento - Classificador Nearest Centroid")
	Nr = 0
	hitsVector = []

	while Nr!=100:
		print("Rodada {}".format(Nr+1))
		X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
		neighCentroid.fit(X_train, y_train)

		count = 0
		predict = neighCentroid.predict(X_test)
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

#Testar função -> nearest_centroid_training()
#centroid_training()

def knearest_rank_a_sample():
	#take a picture for classification
	#sample_capture_to_rank()
	
	#read the image that had captured
	img = load_picture_captured()
	img = img.reshape(1,-1) # for convert in a single sample.
	
	# Normalizar imagem capturada e o dataset
	img = normalize(img)

	neighKNeigh.fit(X,y)
	predict = neighKNeigh.predict(img)
	print("The system predicts :{}".format(predict))

#Testar função -> knearest_rank_a_sample()
knearest_rank_a_sample()

def nearest_centroid_rank_a_sample():
	#take a picture for classification
	#sample_capture_to_rank()
	
	#read the image that had captured
	img = load_picture_captured()
	img = img.reshape(1,-1) # for convert in a single sample.
	
	# Normalizar imagem capturada e o dataset
	img = normalize(img)

	neighCentroid.fit(X,y)
	predict = neighCentroid.predict(img)
	print("The system predicts :{}".format(predict))

#Testar função -> nearest_centroid_rank_a_sample()
nearest_centroid_rank_a_sample()
