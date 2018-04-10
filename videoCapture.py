import numpy as np
import cv2

def video_capture():
	cap = cv2.VideoCapture(0)
	count = 1

	while count == 1:
		ret, frame = cap.read()
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		print("Salvando imagem ...")
		cv2.imwrite("frame.jpg",gray)
		count +=1
	print("Done.")
	#cap.release()
	#cap.destroyAllWindows()


