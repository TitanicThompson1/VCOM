from glob import glob
import cv2
from matplotlib import image

# # Alinea a)

# image_name = input("Enter the image name: ")

# image = cv2.imread(image_name)

# cv2.imshow(image_name, image)

# cv2.waitKey(0)

# cv2.destroyWindow(image_name)

# h, w, _ = image.shape

# print(f"Height: {h}", f"Width: {w}", sep='\n')

# # Alinea b)

# cv2.imwrite("dog.bmp", image)

# Alinea c)


# def read_pixel_color(event, x, y, flags, param):
#     global image

#     if event == cv2.EVENT_MOUSEMOVE:
#         print("Pixel Color: ", image[x, y])

# image = cv2.imread("dog_print.PNG")

# cv2.imshow("dog", image)
# cv2.setMouseCallback("dog", read_pixel_color)

# cv2.waitKey(0)

# cv2.destroyAllWindows()

# Alinea d)

first_point = []
second_point = []
click_number = 0

image = cv2.imread("dog_print.PNG")

cv2.imshow("dog", image)
roi = cv2.selectROI("dog", image)

cropped_image = image[roi[1]:roi[1]+roi[3], roi[0]:roi[0]+roi[2]]

cv2.imwrite("dog_section.png", cropped_image)
cv2.waitKey(0)

cv2.destroyAllWindows()




# def select_two_points(event, x, y, _, _dummy):
#     global image
#     global click_number

#     if event == cv2.EVENT_LBUTTONDOWN:
#         print("Captured first click")
#         if click_number == 0:
#             global first_point
#             first_point = [x, y]
#         else:
#             second_point = [x, y]
#             click_number = 0

#             cv2.selectROI()

