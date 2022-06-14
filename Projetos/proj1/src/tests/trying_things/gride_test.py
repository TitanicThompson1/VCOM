import cv2 as cv


image = cv.imread("dataset/simple_square_blue/yes3.png")
print(image.shape)
h = image.shape[0]
w = image.shape[1]
n = 10
for i in range(1, n):
    cv.line(image, (i * (w // n), 0), (i * (w// n), h), (0, 0, 255), 1)
    cv.line(image, (0, i * (h// n)), (w, i * (h // n)), (0, 0, 255), 1)

cv.imshow("image", image)
cv.waitKey(0)
cv.destroyAllWindows()
