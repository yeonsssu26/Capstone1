import os
import sys
import json
from xml.etree.ElementTree import parse

def get_class(xml_path):
    tree = parse(xml_path)
    root = tree.getroot()
    classes = root.findall("object")
    names = [x.findtext("name") for x in classes]
    return names
    
    

path = sys.argv[1]
files = os.listdir(path)
classlist = []

for file in files:
    classes = get_class(path+'\\'+file)
    classlist = list(set(classlist) | set(classes))

classlist.sort()
f = open("label.txt", 'w')
for ca in classlist:
    f.write(ca+'\n')
f.close()