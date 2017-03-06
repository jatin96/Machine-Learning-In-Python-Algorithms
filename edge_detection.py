import numpy as np
import cv2
cap = cv2.VideoCapture(0)
while True:
	_, frame = cap.read();
	edges = cv2.Canny(frame,100,200)

	cv2.imshow('frame',frame)
	cv2.imshow('edges',edges)
	#waitKey(5);
	k = cv2.waitKey(5) & 0xFF
	if k == ord('q') :
		break
cv2.destroyAllWindows()
cap.release()
