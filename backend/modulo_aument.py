import albumentations as A
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
from bbaug import policies
import cv2
import os
from pascal_voc_writer import Writer
from xml.dom import minidom
import imgaug as ia
import imgaug.augmenters as iaa
import math
import random
import copy
import glob

#imagespath = 'C:/Users/Ruben/python-practice/Object-Detection-Yolo/dataset/frames/train/images/'
random.seed(7)
#directory = r'C:/Users/Ruben/python-practice/Object-Detection-Yolo/dataset/frames/train/images/'


#os.chdir(directory)

def readImage(filename,imagespath):
    # OpenCV uses BGR channels
    img = cv2.imread(os.path.join(imagespath,filename))
    return img

def readYolo(filename):
    coords = []
    with open(filename, 'r') as fname:
        for file1 in fname:
            x = file1.strip().split(' ')
            x.append(x[0])
            x.pop(0)
            x[0] = float(x[0])
            x[1] = float(x[1])
            x[2] = float(x[2])
            x[3] = float(x[3])
            coords.append(x)
    return coords

def getTransform(loop):
    if loop == 0:
        transform = A.Compose([
            A.HorizontalFlip(p=1),
        ], bbox_params=A.BboxParams(format='yolo'))
    elif loop == 1:
        transform = A.Compose([
            A.RandomBrightnessContrast(p=1),
        ], bbox_params=A.BboxParams(format='yolo'))
    elif loop == 2:
        transform = A.Compose([
            A.MultiplicativeNoise(multiplier=0.5, p=0),
        ], bbox_params=A.BboxParams(format='yolo'))
    elif loop == 3:
        transform = A.Compose([
            A.VerticalFlip(p=1)
        ], bbox_params=A.BboxParams(format='yolo'))
    elif loop == 4:
        transform = A.Compose([
            A.Blur(blur_limit=(50, 50), p=0)
        ], bbox_params=A.BboxParams(format='pascal_voc'))
    elif loop == 5:
        transform = A.Compose([
            A.Transpose(1)
        ], bbox_params=A.BboxParams(format='yolo'))
    elif loop == 6:
        transform = A.Compose([
            A.RandomRotate90(p=1)
        ], bbox_params=A.BboxParams(format='yolo'))
    elif loop == 7:
        transform = A.Compose([
            A.JpegCompression(quality_lower=0, quality_upper=1, p=0.2)
        ], bbox_params=A.BboxParams(format='yolo'))
    elif loop == 8:
        transform = A.Compose([
            A.Cutout(num_holes=50, max_h_size=40,
                     max_w_size=40, fill_value=128, p=0)
        ], bbox_params=A.BboxParams(format='pascal_voc'))

    return transform

def writeYolo(coords, count, name):

    with open(name+str(count)+'.txt', "w") as f:
        for x in coords:
            f.write("%s %s %s %s %s \n" % (x[-1], x[0], x[1], x[2], x[3]))

def start(tmp_train_path):            
    count = 3000 #se utilizar para renombrar el archivo Ejemplo IMAGE-13000

    imagespath = os.path.join(tmp_train_path,"images")
    labelspath = os.path.join(tmp_train_path,"labels")

    for filename in os.listdir(imagespath):
        if filename.endswith(".jpg") or filename.endswith(".JPG"):
            title, ext = os.path.splitext(os.path.basename(filename))
            image = readImage(filename,imagespath)

            txt_file_path = os.path.join(labelspath, title + '.txt')
            if os.path.exists(txt_file_path):
            #if filename.endswith(".txt"):
                #xmlTitle, txtExt = os.path.splitext(os.path.basename(filename))
                #if xmlTitle == title:
                    # bboxes = getCoordinates(filename)
                    bboxes = readYolo(txt_file_path)
                    for i in range(0, 9):
                        img = copy.deepcopy(image)
                        transform = getTransform(i)
                        try:
                            transformed = transform(image=img, bboxes=bboxes)
                            transformed_image = transformed['image']
                            transformed_bboxes = transformed['bboxes']
                            name = title+str(count)+'.jpg'
                            cv2.imwrite(os.path.join(imagespath,name) , transformed_image)
                            # print(transformed_bboxes)
                            # writeVoc(transformed_bboxes, count, transformed_image)
                            writeYolo(transformed_bboxes, count,os.path.join(labelspath,title) )
                            count = count+1
                        except:
                            print("bounding box issues")
                            pass

