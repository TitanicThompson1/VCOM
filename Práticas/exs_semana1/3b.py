import cv2

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Can't open camera")
    exit()

frame_nr = 0
while True:
    ret, frame = cap.read()

    if not ret:
        print("Can't receive frame")
        break

    grayscale_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    cv2.imshow('webcam', frame)
    cv2.imshow('cool jazz filter', grayscale_frame)

    if cv2.waitKey(1) == ord('q'):
        break

    frame_nr += 1

cap.release()
cv2.destroyAllWindows()
