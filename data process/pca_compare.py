# encoding=utf-8
import numpy as np
import cv2
import os
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
from sklearn import decomposition
from sklearn.decomposition import PCA
import pandas as pd
from sklearn.preprocessing import StandardScaler

os.environ["CUDA_VISIBLE_DEVICES"] = '1'
# class EigenFace(object):
# def __init__(self, threshold, dimNum):
#     self.threshold = threshold  # 阈值暂未使用
#
#     self.dimNum = dimNum


def loadImg(fileName):
    '''''
    载入图像，灰度化处理，统一尺寸，直方图均衡化
    :param fileName: 图像文件名
    :return: 图像矩阵
    '''

    img = cv2.imread(fileName)
    retImg = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    retImg = cv2.equalizeHist(retImg)

    # cv2.imshow('img',retImg)
    # cv2.waitKey()
    return retImg

def createImgMat(dirName):
    '''''
    生成图像样本矩阵，组织形式为行为属性，列为样本
    :param dirName: 包含训练数据集的图像文件夹路径
    :return: 样本矩阵，标签矩阵
    '''

    dataMat = np.zeros((1444, 1))
    label = []
    for parent, dirnames, filenames in os.walk(dirName):

        index = 0
        for dirname in dirnames:
            # print('dirname',dirname)
            dirpath = os.path.join(parent,dirname)
            # print('path',dirpath)
            for subParent, subDirName, subFilenames in os.walk(dirpath):
                # print('subFilenames',subFilenames)
                for ssubDirName in subDirName:
                    path = os.path.join(dirpath,ssubDirName)
                    for SubParent,SubDirName,SubFilenames in os.walk(path):

                        for filename in SubFilenames:
                            if filename == "part1.jpg":
                                img = loadImg(SubParent + '/' + filename)
                                # print('filename!!!',filename,'img',img)
                                tempImg = np.reshape(img, (-1, 1))
                                if index == 0:
                                    dataMat = tempImg
                                    # print('datamat shape',dataMat.shape)
                                else:
                                    dataMat = np.column_stack((dataMat, tempImg))
                                index += 1
                        label.append(dirnames)

    return dataMat

def createImgMattest(filename):
    img = loadImg(filename)
    img = cv2.resize(img, (25, 25), interpolation=cv2.INTER_CUBIC)
    dataMat = np.reshape(img, (-1, 1))
    return dataMat


if __name__ == '__main__':

    # print createImgMat('/home/hszc/Desktop/ZHIHUICHENGSHI/version3/preprocess/2/test/test/for')
    data=createImgMat('/home/hszc/Desktop/ZHIHUICHENGSHI/version3/preprocess/2/test/test/for')



    # estimator = PCA(n_components=400)\
    sc=StandardScaler()
    data_std=sc.fit_transform(data.T)

    # pca= decomposition.PCA(n_components=900,copy=True)
    pca = PCA(n_components=625, svd_solver='randomized',
              whiten=True)
    # pca = PCA(n_components=900, svd_solver='randomized',
              # whiten=True)
    n_samples, n_features = data_std.shape
    # print('n_samples, n_features',n_samples, n_features)
    # print("before",data_std.shape)

    data_rec=pca.fit_transform(data_std)
    print("after",data_rec.shape)

    print('data shape',data_rec.shape)
    # the mean face
    finalface = np.mat(np.mean(data_rec, 0)).T
    print('finalface shape',finalface.shape)
    # meanface = np.reshape(finalface, (25,25 ))
    # cv2.imwrite('/home/hszc/Desktop/ZHIHUICHENGSHI/version3/preprocess/2/test/test/{}.jpg'.format(str('meanface1')),meanface)
    threshold = 380
    #testdata
    testdir='/home/hszc/Desktop/ZHIHUICHENGSHI/version3/preprocess/2/test/part/unnormalface/'
    for partfilepath, dirnames, filenames in os.walk(testdir):
       if filenames:
            for filename in filenames:
                datapath='/home/hszc/Desktop/ZHIHUICHENGSHI/jiayou/data/meanface/part1/'+filename


                # meanfaceMat is finalface
                data = finalface
                testpath=partfilepath+'/'+filename
                testdata = createImgMattest(testpath)

        #
                # diffMat = testdata - data
                diffMat = np.sqrt(np.sum(np.square(testdata - data)))
                print('dir:',partfilepath)
                print('file:',filename)
                print('datamat',diffMat)
                #
                print('data',data.shape)
                print('test',testdata.shape)

                # if diffMat > threshold:
                    # os.remove(testpath)
                    # print(testpath ,'was removed!')
