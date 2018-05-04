import pandas as pd
import numpy as np
import cv2

import subprocess

# Saudações do sistema
#subprocess.call(["say","Olá, seja bem vindo ao sistema de reconhecimento facial do Centauro"])
#subprocess.call(["say","Minha missão, pelo menos por enquanto, é tentar reconhecer você"])

from nearestNeighborsClassifier import knearest_neighborhood_training,centroid_training
#Verificar taxas de acertos, treinamento e teste dos algorítmos
#knearest_neighborhood_training()
#centroid_training()


from nearestNeighborsClassifier import knearest_rank_a_sample,nearest_centroid_rank_a_sample

# Classificar uma amostra utilizando o classificador vizinho mais próximo
#knearest_rank_a_sample()

# Classificar uma amostra utilizando o classificador centroid mais próximo
#nearest_centroid_rank_a_sample()


from imageCapture import sample_capture_to_rank,samples_capture_to_dataBase

# Função para salvar novas imagens no Banco de Dados
#samples_capture_to_dataBase(8,5)


from loadFaces import vectorize_data_faces
from saveReturnValuesCSV import save_vectorized_load_faces_in_csv_file

#save_vectorized_load_faces_in_csv_file(vectorize_data_faces('samples_faces_dataset',7))