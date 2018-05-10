import cv2
from detectFaces import detect_faces,track_face

f_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')

video_capture = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()
    
    img_copy = frame.copy()          

    gray = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)
    
    faces = f_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5);          

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()