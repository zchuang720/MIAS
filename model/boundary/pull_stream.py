import cv2
import numpy as np

src = "http://192.168.65.202:8080/live/monitoring/2.flv"

cap = cv2.VideoCapture(src)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    cv2.imshow("frame", cv2.resize(frame, [800,600]))
    if cv2.waitKey(50) & 0xFF == ord('q'):
        break