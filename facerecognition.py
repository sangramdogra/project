import cv2
import numpy as np
import matplotlib.pyplot as plt
import cvlib as cv 

#Testing model to capture face recognition and output via html
capture = cv2.VideoCapture(0)
while True:
    ret, frame = capture.read()
    faces, conf = cv.detect_face(frame)
    if faces != []:
        for face in faces:
            frame = cv2.rectangle(frame, (face[0], face[1]), (face[2], face[3]), (255, 0, 0))
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
capture.release()
# Destroy all the windows
cv2.destroyAllWindows()
