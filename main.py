import pandas as pd
import numpy as np
import subprocess
import cv2
import matplotlib.pyplot as plt
from imageCapture import sample_capture_to_rank,samples_capture_to_dataBase
from loadFaces import detect_faces,vectorize_data_faces_cutting
from nearestNeighborsClassifier import knearest_rank_a_sample,nearest_centroid_rank_a_sample

# Saudações do sistema
#subprocess.call(["say","Olá, seja bem vindo ao sistema de reconhecimento facial do Centauro"])
#subprocess.call(["say","Minha missão, pelo menos por enquanto, é tentar reconhecer você"])

# Classificar utilizando o classificador vizinho mais próximo
#knearest_rank_a_sample()

# Classificar utilizando o classificador centroid mais próximo
nearest_centroid_rank_a_sample()
