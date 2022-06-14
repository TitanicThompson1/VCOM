import cv2 as cv
from pathlib import Path
import red_circles
from progress.bar import Bar


path = Path('dataset/formatted_images')
files = [str(x) for x in path.glob('*.png')]

log = ""
true_positives = 0
false_positives = 0
false_negatives = 0
true_negatives = 0


with Bar('Processing...', max=len(files)) as bar:
    
    for file in files:
        print(file)
        # print("Testing file:", file)
        image = cv.imread(file)
        if image is None:
            print(f'Image {file} not loaded')
            continue
        
        if "sp" in file:
            if red_circles.find_red_circles_signs(image):
                true_positives += 1
                log += f"TP on {file}\n"
            else:
                false_negatives += 1
                log += f"FN on {file}\n"
        else:
            if red_circles.find_red_circles_signs(image):
                false_positives += 1
                log += f"FP on {file}\n"
            else:
                true_negatives += 1
                log += f"TN on {file}\n"

        bar.next()

# write to file
with open("logs/s_ef_testcircles_5.txt", "w") as f:
    f.write(log)
    f.write(f"True positives: {true_positives}\n")
    f.write(f"False positives: {false_positives}\n")
    f.write(f"False negatives: {false_negatives}\n")
    f.write(f"True negatives: {true_negatives}\n")

    f.write(f"Total: {true_positives + false_positives + false_negatives + true_negatives}\n")
    f.write(f"Accuracy: {(true_positives + true_negatives) / (true_positives + false_positives + false_negatives + true_negatives)}\n")
    f.write(f"Precision: {true_positives / (true_positives + false_positives)}\n")
    f.write(f"Recall: {true_positives / (true_positives + false_negatives)}\n")
