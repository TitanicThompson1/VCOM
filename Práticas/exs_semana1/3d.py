import cv2 as cv
import numpy as np

cap = cv.VideoCapture(0)

if not cap.isOpened():
    print("Can't open camera")
    exit()

frame_nr = 0
while True:
    ret, frame = cap.read()

    if not ret:
        print("Can't receive frame")
        break

    hsv_frame = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    binary_frame = cv.inRange(hsv_frame, np.array([0, 0, 0]), np.array([20, 255, 255]))
    # _, binary_frame = cv.threshold(hsv_frame, 20, 128, cv.THRESH_BINARY)
    tracked_obj = cv.bitwise_and(frame, np.pad(binary_frame, (0, 2), 'constant'))

    cv.imshow('webcam', frame)
    cv.imshow('Tracker', tracked_obj)

    if cv.waitKey(1) == ord('q'):
        break

    frame_nr += 1

cap.release()
cv.destroyAllWindows()
