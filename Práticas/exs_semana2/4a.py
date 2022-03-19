import cv2
import os

dataDir = '../Images_01'

img = cv2.imread(os.path.join(dataDir, 'coins_03_noisy.jpg'))

cv2.imshow("Coin", img)

img_blured = cv2.blur(img, (7,7))

cv2.imshow("Blured Coin", img_blured)

cv2.waitKey(0)

cv2.destroyAllWindows()
