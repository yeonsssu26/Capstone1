import os
import sys
import json

# path = sys.argv[1]

# files = os.listdir(path)

# f = open("ids.txt", 'w')
# for filename in files:
#     filename = filename.rstrip('.xml')
#     f.writelines(filename+"\n")
# f.close()

f = open("ids2.txt", 'w')
for i in range(200):
    f.write(str(i)+'\n')
f.close()