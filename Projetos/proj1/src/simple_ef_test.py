import cv2 as cv
import red_circles
from square_blue_signs import find_square_blue_signs
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
    
    if "bs" in file or "cw" in file:
        if len(find_square_blue_signs(image)) > 0:
            true_positives += 1
            log += f"TP on {file}\n"
        else:
            false_negatives += 1
            log += f"FN on {file}\n"
    else:
        if len(find_square_blue_signs(image)) > 0:
            false_positives += 1
            log += f"FP on {file}\n"
        else:
            true_negatives += 1
            log += f"TN on {file}\n"

        # bar.next()

# write to file
with open("logs/final_test.txt", "w") as f:
    f.write(log)
    f.write(f"True positives: {true_positives}\n")
    f.write(f"False positives: {false_positives}\n")
    f.write(f"False negatives: {false_negatives}\n")
    f.write(f"True negatives: {true_negatives}\n")

    f.write(f"Total: {true_positives + false_positives + false_negatives + true_negatives}\n")
    f.write(f"Accuracy: {(true_positives + true_negatives) / (true_positives + false_positives + false_negatives + true_negatives)}\n")
    f.write(f"Precision: {true_positives / (true_positives + false_positives)}\n")
    f.write(f"Recall: {true_positives / (true_positives + false_negatives)}\n")
