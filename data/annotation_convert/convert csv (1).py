import os
import csv
import xlsxwriter
from PIL import Image
import re

# make empty csv file
workbook = xlsxwriter.Workbook("dataset.xlsx")
worksheet = workbook.add_worksheet("sheet1")

# resize col
worksheet.set_column("A:A", 60)

# get current directory path
directory = os.getcwd()

img_file_name = []
text_file_name = []
for filename in os.listdir(directory):
    if filename.endswith(".png"):
        path = os.path.join(directory, filename)
        img_file_name.append(path)

    elif filename.endswith(".txt"):
        path = os.path.join(directory, filename)
        text_file_name.append(path)

print(img_file_name)  # for check
print(text_file_name)  # for check

# text annotation
text = []
for t in text_file_name:
    plate = []
    with open(t, "r") as f:
        for line in f:
            if line[:5] == "plate":
                plate.append(line.strip("plate: ").strip())

            elif line[:14] == "position_plate":
                numbers = list(map(int, line.strip("position_plate: ").strip().split()))
                plate.append(numbers)

            elif "char" in line:
                numbers = re.findall("\d+", line[8:])
                plate.append(list(map(int, numbers)))
    text.append(plate)
print(text)  # for check

#################################################################################
#################### error : 마지막 image, text에 대해서만 됨 ####################
#################################################################################
# insert to csv
idx = 0
k = 0
for filename in img_file_name:
    # col A, B
    for i in range(k, k + len(text[idx][0]) - 1):
        A = "A" + str(i + 1)
        path = os.path.join(directory, filename)
        worksheet.write(A, path)

        B = "B" + str(i + 1)
        worksheet.write(B, text[idx][0])

    k += len(text[idx][0]) - 1
    idx += 1

idx = 0
k = 0
for filename in text_file_name:
    # col C, D, E, F : x1, y1, x2, y2
    j = 0
    for i in range(k, k + len(text[0]) - 2):
        C = "C" + str(i + 1)
        worksheet.write(C, text[idx][j + 2][0])

        D = "D" + str(i + 1)
        worksheet.write(D, text[idx][j + 2][1])

        E = "E" + str(i + 1)
        worksheet.write(E, text[idx][j + 2][0] + text[idx][j + 2][2])

        F = "F" + str(i + 1)
        worksheet.write(F, text[idx][j + 2][1] + text[idx][j + 2][3])

        j += 1
    k += len(text[0]) - 2
    idx += 1

workbook.close()
