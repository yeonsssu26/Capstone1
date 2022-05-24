import cv2
import os
import sys
import re
import glob

path = sys.argv[1]
save_path = sys.argv[2]

def cropFolerlevel(path, save_path):
    img_list = [file for file in glob.glob(path+'\\*.png')]
    txt_list = [file for file in glob.glob(path+'\\*.txt')]

    for i in range(img_list.__len__()):
        txt = open(txt_list[i])
        img = cv2.imread(img_list[i])
        name = img_list[i].split('\\')[-1]
        
        p = list(map(int, re.split(' |\n', txt.readlines()[7])[1:5]))
        crop_img = img[p[1]:p[1]+p[3], p[0]:p[0]+p[2]]
        cv2.imwrite(save_path+'\\'+name , crop_img)

def recursive(path, save_path):
    files = os.listdir(path)
    for file in files:
        fullname = os.path.join(path, file)
        savename = os.path.join(save_path, file)
        if os.path.isdir(fullname):
            recursive(fullname, savename)
    if not os.path.isdir(save_path):
        os.makedirs(save_path)
    cropFolerlevel(path, save_path)

recursive(path, save_path)