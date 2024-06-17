import cv2, glob
import numpy as np
height=64
file_name="./output.txt"
file=open(file_name)
lines=file.readlines()
pic=np.zeros((1000,1000))
for idx1,line in enumerate(lines):
    pixels=line.split(" ")
    for idx2,pixel in enumerate(pixels):
        if pixel!="\n":
            pic[idx1,idx2]=int(float(pixel))

cv2.imwrite("./see.png",pic)
