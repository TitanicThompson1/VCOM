import cv2
import os

dataDir = '../Images_01'

img = cv2.imread(os.path.join(dataDir, 'coins_03_noisy.jpg'))

cv2.imshow("Coin", img)

for i in range(1, 9, 2):

    img_blured = cv2.medianBlur(img, i)

    cv2.imshow(f"Blured Coin {i}", img_blured)

cv2.waitKey(0)

cv2.destroyAllWindows()
