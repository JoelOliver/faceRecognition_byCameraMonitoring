import numpy as np
import cv2
from loadFaces import vectorize_data_faces
from saveReturnValuesCSV import save_vectorized_load_faces_in_csv_file,return_last_index_subject
import subprocess
from loadFaces import vectorize_data_faces_cutting,vectorize_data_faces
from detectFaces import detect_faces

# Função para capturar apenas uma imagem, para que o programa a classifique
def sample_capture_to_rank():
	#subprocess.call(["say","Para reconhecer você, será necessário que você se posicione em frente a camêra"])
	#subprocess.call(["say","Uma janela será aberta, e, quando estiver preparado aperte a tecla espaço para continuar!"])

	cam=cv2.VideoCapture(0)
	cv2.namedWindow("image_capture",cv2.WINDOW_NORMAL)
	cv2.resizeWindow('image_capture', 600,600)

	print(">>> Pressione a tecla SPACE para capturar a imagem ou ESC para sair <<< ")
	haar_face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
	while True:
		ret, frame = cam.read()
		cv2.imshow('image_capture', frame)
		if not ret:
			break
		k = cv2.waitKey(1)

		if k%256 == 27:
			#ESC pressed
			print("ESC apertado, fechando a janela ...")
			break
		elif k%256 == 32:
			#SPACE pressed
			img_name = 'sample_to_rank.png'
			try:
				faces_detected_img = detect_faces(haar_face_cascade, frame)
				cv2.imwrite(img_name,cv2.cvtColor(faces_detected_img, cv2.COLOR_BGR2GRAY))
				print('captura realizada com sucesso !')
				break 
			except:
				print('A captura não foi possível, por favor se posicionar adequadamente em frente a câmera e apertar a tecla ESPAÇO quando estiver preparado ...')

	cam.release()

	cv2.destroyAllWindows()

#Testar função -> sample_capture_to_rank()
#sample_capture_to_rank()

def samples_capture_to_dataBase(subject_number):
	cam=cv2.VideoCapture(0)
	cv2.namedWindow("image_capture_for_database",cv2.WINDOW_NORMAL)
	cv2.resizeWindow("image_capture_for_database",800,800)

	print("Capturar cerca de 10 fotos para que componham o Banco de Dados,")
	print("Basta utilizar a tecla ESPAÇO para que a captura seja possível realizar uma captura ...")

	subject_number+=1
	npictures=0
	while npictures <10:
		ret,frame = cam.read()
		cv2.imshow('image_capture_for_database',frame)
		if not ret:
			break
		k= cv2.waitKey(1)

		if k%256 == 27:
			#ESC pressed
			print("ESC apertado, fechando a janela ...")
			break
		elif k%256 == 32:
			#SPACE pressed
			img_name = 'samples_faces_dataset/Subject_{}_type_{}.png'.format(subject_number,npictures+1)
			try:
				faces_detected_img = detect_faces(haar_face_cascade, frame)
				cv2.imwrite(img_name,cv2.cvtColor(faces_detected_img, cv2.COLOR_BGR2GRAY))
				print('{} salvo'.format(img_name))
				npictures+=1
			except:
				print('A captura não foi possível, por favor se posicionar adequadamente em frente a câmera e apertar a tecla ESPAÇO quando estiver preparado ...')

	cam.release()

	cv2.destroyAllWindows()