import xml.etree.ElementTree as ET
import os
 
xmlRootDir_train = 'VOC2012_train/Annotations/'
files_train = os.listdir(xmlRootDir_train)
#xmlRootDir_test = 'VOC2012_test/Annotations/'
#files_test = os.listdir(xmlRootDir_test)
 
def parseXML(filename):
    bbox = [[],[],[],[],[]]
    tree = ET.parse(filename)
    root = tree.getroot()
    size = root.find('size')
    width = float(size.find('width').text)
    height = float(size.find('height').text)
    for node in root.iter("object"):
        bndbox = node.find('bndbox')
        classname = node.find('name').text
        try:
            xmin = int(bndbox.find('xmin').text)
            ymin = int(bndbox.find('ymin').text)
            xmax = int(bndbox.find('xmax').text)
            ymax = int(bndbox.find('ymax').text)
            bbox[0].append(classname)
            bbox[1].append(xmin)
            bbox[2].append(ymin)
            bbox[3].append(xmax)
            bbox[4].append(ymax)
        except:
            print(filename)
    return bbox
 
bboxfile = open('bbox_train.csv', 'w')
content = ''
#i = 0
for xmlfile in files_train:
    bbox = parseXML(xmlRootDir_train+xmlfile)
    content += xmlfile
    for j in range(5):
        content += ','+';'.join([str(x) for x in bbox[j]])
    content += '\n'
'''
for xmlfile in files_test:
    bbox = parseXML(xmlRootDir_test+xmlfile)
    content += xmlfile
    for j in range(5):
        content += ','+';'.join([str(x) for x in bbox[j]])
    content += '\n'
'''
#print("processing %i/1000\r"%i, end="")
bboxfile.writelines(content)
bboxfile.close()
