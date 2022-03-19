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

    cv2.imshow('webcam', frame)

    if cv2.waitKey(1) == ord('s'):
        frame_name = 'frame' + str(frame_nr) + ".png"
        cv2.imwrite(frame_name, frame)
        cv2.imshow("Saved frame: " + frame_name, frame)

        cv2.waitKey(0)
        cv2.destroyWindow("Saved framea: " + frame_name)

    if cv2.waitKey(1) == ord('q'):
        break

    frame_nr += 1

cap.release()
cv2.destroyAllWindows()

    