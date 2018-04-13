import numpy as np
import cv2

def image_capture():
	cam=cv2.VideoCapture(0)

	cv2.namedWindow("test",cv2.WINDOW_NORMAL)

	cv2.resizeWindow('test', 600,600)

	while True:
		ret, frame = cam.read()
		cv2.imshow('test', frame)
		if not ret:
			break
		k = cv2.waitKey(1)

		if k%256 == 27:
			#ESC pressed
			print("Escape hit, closing ...")
			break
		elif k%256 == 32:
			#SPACE pressed
			img_name = 'frame_sample.png'
			cv2.imwrite(img_name,cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
			print('{} written !'.format(img_name))
			break 

	cam.release()

	cv2.destroyAllWindows()


