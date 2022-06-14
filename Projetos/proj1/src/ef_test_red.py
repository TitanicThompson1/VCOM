import cv2 as cv
import red_circles
from square_blue_signs import find_square_blue_signs
from stop_sign import find_stop_signs
from pathlib import Path

path = Path('dataset/formatted_images')
files = [str(x) for x in path.glob('*.png')] + [str(x) for x in path.glob('*.jpeg')]

log = ""
true_positives = 0
false_positives = 0
false_negatives = 0
true_negatives = 0
    
for file in files:

    image = cv.imread(file)
    if image is None:
        print(f'Image {file} not loaded')
        continue
    
    stop_signs = find_stop_signs(image)
    red_signs = red_circles.find_red_circles(image)
    blue_signs = find_square_blue_signs(image)
    if ("st" in file and len(stop_signs) > 0) or (("cw" in file or "bs" in file) and len(blue_signs) > 0) or ("sp" in file and len(red_signs) > 0):
        good += 1
    else:


        # bar.next()

# write to file
with open("logs/final_test_stop.txt", "w") as f:
    f.write(log)
    f.write(f"True positives: {true_positives}\n")
    f.write(f"False positives: {false_positives}\n")
    f.write(f"False negatives: {false_negatives}\n")
    f.write(f"True negatives: {true_negatives}\n")

    f.write(f"Total: {true_positives + false_positives + false_negatives + true_negatives}\n")
    f.write(f"Accuracy: {(true_positives + true_negatives) / (true_positives + false_positives + false_negatives + true_negatives)}\n")
    f.write(f"Precision: {true_positives / (true_positives + false_positives)}\n")
    f.write(f"Recall: {true_positives / (true_positives + false_negatives)}\n")
