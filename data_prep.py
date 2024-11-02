import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import cv2
import os

train = "./archive/MineCraft-RT_1280x720_v14/MineCraft-RT_1280x720_v14"
working_dir = './outputs'
print("Preparing Training Data")
for dirname, _, filenames in os.walk(train + '/images'):
    for filename in filenames:
        label = cv2.imread(dirname +'/'+ filename)
        input = cv2.resize(label, (0,0), fx=0.25, fy=0.25) 
        cv2.imwrite(working_dir+'/train/inputs/in_' + filename, input)
        
val = "./archive/MineCraft-RT_1280x720_v12/MineCraft-RT_1280x720_v12"
print("Preparing Val Data")
for dirname, _, filenames in os.walk(val + '/images'):
    for filename in filenames:
        label = cv2.imread(dirname +'/'+ filename)
        input = cv2.resize(label, (0,0), fx=0.25, fy=0.25) 
        cv2.imwrite(working_dir+'/val/inputs/in_' + filename, input)

