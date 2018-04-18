import numpy as np
import cv2
from loadFaces import vectorize_data_faces,return_last_index_subject,save_vectorized_load_faces_in_csv_file

# Função para capturar apenas uma imagem, para que o programa classifique
# a partir da escolha do classificador-algorítmo e Banco de Dados
def sample_capture_to_rank():

	cam=cv2.VideoCapture(0)
	cv2.namedWindow("image_capture",cv2.WINDOW_NORMAL)
	cv2.resizeWindow('image_capture', 600,600)

	print("Pressione a tecla ESC para sair ou SPACE para capturar a imagem")

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
			cv2.imwrite(img_name,cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
			print('captura realizada !')
			break 

	cam.release()

	cv2.destroyAllWindows()

#Testar função -> sample_capture_to_rank()
#sample_capture_to_rank()

def samples_capture_to_dataBase(subject_number):
	cam=cv2.VideoCapture(0)
	cv2.namedWindow("image_capture_for_database",cv2.WINDOW_NORMAL)
	cv2.resizeWindow("image_capture_for_database",800,800)

	print("Tirar cerca de 10 fotos para que componham o Banco de Dados,")
	print("Basta utilizar o SPACE ...")

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
			cv2.imwrite(img_name,cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
			print('{} salvo'.format(img_name))
			npictures+=1

	cam.release()

	cv2.destroyAllWindows()
	v=vectorize_data_faces('samples_faces_dataset', return_last_index_subject())
	save_vectorized_load_faces_in_csv_file(v)

#Usar função -> samples_capture_to_dataBase()
#samples_capture_to_dataBase(return_last_index_subject())