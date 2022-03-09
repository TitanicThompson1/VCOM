from calendar import c
import numpy as np
import cv2

image = cv2.imread("dog_print.PNG")

row, col, ch = image.shape
s_vs_p = 0.5
amount = 0.1
out = np.copy(image)
print(image.size)
# Salt mode
num_salt = np.ceil(amount * image.size * s_vs_p)
coords = [np.random.randint(0, i - 1, int(num_salt))
          for i in image.shape[0:2]]

out[coords] = (255, 255, 255)

# Pepper mode
num_pepper = np.ceil(amount * image.size * (1 - s_vs_p))
coords = [np.random.randint(0, i - 1, int(num_pepper))
          for i in image.shape[0:2]]

out[coords] = (0, 0, 0)

cv2.imshow("seasoned dog", out)

cv2.waitKey(0)

cv2.destroyAllWindows()