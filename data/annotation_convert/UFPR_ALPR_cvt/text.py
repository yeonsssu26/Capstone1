import cv2
import os
import sys
import glob
import re

path = sys.argv[1]
save_path = sys.argv[2]
img_list = [cv2.imread(file) for file in glob.glob(path + '\\*.png')]
txt_list = [file for file in glob.glob(path + '\\*.txt')]

print(cv2.__version__)
print(txt_list)
print(img_list[1])
txt = open(txt_list[1])
img = img_list[1]
p = list(map(int, re.split(' |\n', txt.readlines()[7])[1:5]))
print(p)
cv2.imshow('hello', img)
cv2.waitKey(0)
corp_img = img[p[1]:p[1]+p[3], p[0]:p[0]+p[2]]
cv2.imshow('hello', corp_img)
cv2.imwrite(save_path+'\\result.png', corp_img)
