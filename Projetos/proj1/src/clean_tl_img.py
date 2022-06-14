import os
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

translation = {
    'speedlimit' : 'sp',
    'stop': 'st',
    'trafficlight': 'tl',
    'crosswalk': 'cw',
}
counter = 1

image_path = 'dataset/formatted_images/'

# Read all files from directory
path = Path(image_path)
files = [str(x) for x in path.glob('*.png')]

for file in files:
    if 'tl' in file:
        # Remove the file
        print(f'Removing {file}')
        os.remove(file)

