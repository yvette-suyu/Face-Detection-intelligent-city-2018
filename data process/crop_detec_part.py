import tensorflow as tf
import detect_util
import numpy as np
import os
import time
import cv2
import math
from PIL import Image

landmark=[]
www=[]
hhh=[]
bbbox=[]
"""
...  some codes are left~~~
"""
color_value=[]

def colorvalue(im_path):
    """

    :param imagepath__: each image path,including part image
    :return:
    """
    imageData = Image.open(im_path)
    imageWidth, imageHeight = imageData.size  # height = vertikal, width = horizontal
    RGBImagePixels = imageData.load()
    for yCoordinate in range(0, imageHeight):
        for xCoordinate in range(0, imageWidth):
            redPixelValue, greenPixelValue, bluePixelValue = RGBImagePixels[xCoordinate, yCoordinate]
    """ Compute average color value according to the image block's colorspace """

    sumOfRedPixelValue = 0
    sumOfGreenPixelValue = 0
    sumOfBluePixelValue = 0
    for yCoordinate in range(0, imageHeight):  # compute sum of the pixel value
        for xCoordinate in range(0, imageWidth):
            tmpR, tmpG, tmpB = RGBImagePixels[xCoordinate, yCoordinate]
            sumOfRedPixelValue += tmpR
            sumOfGreenPixelValue += tmpG
            sumOfBluePixelValue += tmpB


    sumOfPixels = imageHeight * imageWidth
    average = (sumOfRedPixelValue + sumOfGreenPixelValue + sumOfBluePixelValue) / sumOfPixels
    sumOfRedPixelValue = sumOfRedPixelValue / (sumOfPixels)  # mean from each of the colorspaces
    sumOfGreenPixelValue = sumOfGreenPixelValue / (sumOfPixels)
    sumOfBluePixelValue = sumOfBluePixelValue / (sumOfPixels)


    color_value.append(sumOfRedPixelValue)
    color_value.append(sumOfGreenPixelValue)
    color_value.append(sumOfBluePixelValue)
    color_value.append(average)
    # print(color_value)


def croppart(landmark,filepath):
    """

    :param landmark: key points for crop localization
    :return:
    """

    part1 = im[0:int(math.ceil(landmark[0][1]) + hhh[0] / 4), 0:int(math.ceil(landmark[0][0]) + www[0] / 4)]
    part2 = im[0:int(math.ceil(landmark[1][1]) + hhh[0] / 4), int(math.ceil(landmark[1][0]) - www[0] / 4):]
    part3 = im[int(math.ceil(landmark[0][1])):int(math.ceil(landmark[3][1])),
            int(math.ceil(landmark[0][0])):int(math.ceil(landmark[1][0]))]
    part4 = im[int(math.ceil(landmark[3][1]) - hhh[0] / 4):, 0:int(math.ceil(landmark[3][0]) + www[0] / 4)]
    part5 = im[int(math.ceil(landmark[4][1]) - hhh[0] / 4):, int(math.ceil(landmark[4][0]) - www[0] / 4):]
    cv2.imwrite(filepath+'part1.png', part1)
    cv2.imwrite(filepath+'part2.png', part2)
    cv2.imwrite(filepath+'part3.png', part3)
    cv2.imwrite(filepath+'part4.png', part4)
    cv2.imwrite(filepath+'part5.png', part5)

    # cv2.imwrite('/home/hszc/Desktop/ZHIHUICHENGSHI/version3/preprocess/2/part5.png', part5)

if __name__ == '__main__':
    from glob import glob
    model_path = '/home/hszc/Desktop/ZHIHUICHENGSHI/version3/preprocess/model/mtcnn'
    im_path = '/home/hszc/Desktop/ZHIHUICHENGSHI/version3/preprocess/2/99.png'
    filepath='/home/hszc/Desktop/ZHIHUICHENGSHI/version3/preprocess/2/part/'
    detector = Detector(model_path, gpu_fraction=0.5)
    # im = cv2.imread(im_path)[:,:,::-1]
    im = cv2.imread(im_path)[:, :, :]
    results = detector.detect_face(im, debug=True)

    if not os.path.exists(filepath):
        os.makedirs(filepath)
    else:
        print('it is existed')
    print('crop to parts')
    croppart(landmark,filepath)
    # original image color
    print('here cv')
    ori_acv=[]
    ori_acv.append(colorvalue(im_path))
    print('original is',ori_acv)

    rootdir = filepath
    list = os.listdir(rootdir)
    imgchara = []
    for i in range(0, len(list)):

        path = os.path.join(rootdir, list[i])
        print(list[i])
        if os.path.isfile(path):
            color_value.append(colorvalue(path))
        imgchara.append(color_value)
        color_value = []
    print('chara',imgchara)




    # print()

