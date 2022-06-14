# Read a text file
import os
import glob
import shutil

lines = []
with open('proj2/src/dataset/train.txt') as f:
    lines = f.readlines()

#:-1 tira o \n
lines = [line[:-1] for line in lines]

for file in lines:
    shutil.copy('proj2/src/dataset/annotations/' + file + '.xml', 'proj2/src/dataset/ann_train/')
