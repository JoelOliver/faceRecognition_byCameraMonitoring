from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn.model_selection import train_test_split
from loadFaces import vectorize_data_faces,load_picture_captured
from saveReturnValuesCSV import return_of_image_and_rotule_vectors,return_subject_name
import numpy as np
from sklearn.preprocessing import normalize
from imageCapture import sample_capture_to_rank
import subprocess

#inicializations
dataset_faces = return_of_image_and_rotule_vectors()
X, y = [dataset_faces[0],dataset_faces[1]]

#print(len(X[0]))

#Normalization of X Matrix of image_vectors
#X = normalize(X)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)

#PCA - Test
from sklearn.decomposition import PCA, IncrementalPCA
#n_components = 9000
#ipca = IncrementalPCA(n_components=n_components)
#X = ipca.fit_transform(X)

#pca = PCA(n_components=n_components)
#X = pca.fit_transform(X)


# para mudar valores dos parâmetros, verificar documentação do scikit-learn
neighKNeigh = KNeighborsClassifier(n_neighbors=3)
neighCentroid = NearestCentroid()

def knearest_neighborhood_training():
	print("\n>>>>> Rodadas de treinamento - Classificador KNearest Neighborhood <<<<<")
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
	
	neighKNeigh.fit(X_train, y_train)

	predictions = neighKNeigh.predict(X_test)

	from sklearn.metrics import classification_report,confusion_matrix
	print(classification_report(y_test,predictions))
	
	'''
	Nr = 0
	hitsVector = []

	while Nr != 100:
		#print("Rodada {}".format(Nr+1))
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
	print("Maximo acerto :{} % \n".format(np.max(np.array(hitsVector))))

	'''

#Testar função -> nearest_knearest_neighborhood_training() 
#knearest_neighborhood_training() 

def centroid_training():
	print("\n>>>>> Rodadas de treinamento - Classificador Nearest Centroid <<<<<")
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
	
	neighCentroid.fit(X_train, y_train)

	predictions = neighCentroid.predict(X_test)

	from sklearn.metrics import classification_report,confusion_matrix
	print(classification_report(y_test,predictions))
	
	'''
	Nr = 0
	hitsVector = []

	while Nr!=100:
		#print("Rodada {}".format(Nr+1))
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
	print("Maximo acerto :{} % \n".format(np.max(np.array(hitsVector))))
	'''

#Testar função -> nearest_centroid_training()
#centroid_training()

def knearest_rank_a_sample(voice=False):

	#take a picture for classification
	sample_capture_to_rank()
	
	#read the image that had captured
	img = load_picture_captured()
	img = img.reshape(1,-1) # for convert in a single sample.
	
	# Normalizar imagem capturada e o dataset
	img = normalize(img)

	neighKNeigh.fit(X,y)
	predict = neighKNeigh.predict(img)
	
	if voice:
		subprocess.call(["say","Olá {}, espero ter acertado.".format(return_subject_name(predict))])
	print("\nO Sistema prediz que você é : {}-{}\n".format(predict,return_subject_name(predict)))

#Testar função -> knearest_rank_a_sample()
#knearest_rank_a_sample()

def nearest_centroid_rank_a_sample(voice=False):
	
	#take a picture for classification
	sample_capture_to_rank()
	
	#read the image that had captured
	img = load_picture_captured()
	img = img.reshape(1,-1) # for convert in a single sample.
	
	# Normalizar imagem capturada e o dataset
	#img = normalize(img)

	#Aplicar PCA - Testando ...
	#img = pca.fit_transform(img)

	neighCentroid.fit(X,y)
	predict = neighCentroid.predict(img)
	
	if voice:
		subprocess.call(["say","Olá {}, espero ter acertado.".format(return_subject_name(predict)[0])])
	print("\nO Sistema prediz que você é : {}-{}\n".format(predict,return_subject_name(predict)))

#Testar função -> nearest_centroid_rank_a_sample()
#nearest_centroid_rank_a_sample()
