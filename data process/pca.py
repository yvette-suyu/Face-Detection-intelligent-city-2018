# encoding=utf-8 
import numpy as np
import cv2
import os


class EigenFace(object):
    def __init__(self, threshold, dimNum, dsize):
        self.threshold = threshold  # 阈值暂未使用 

        self.dimNum = dimNum
        self.dsize = dsize

    def loadImg(self, fileName):
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

    def createImgMat(self, dirName):
        ''''' 
        生成图像样本矩阵，组织形式为行为属性，列为样本 
        :param dirName: 包含训练数据集的图像文件夹路径 
        :return: 样本矩阵，标签矩阵 
        '''

        dataMat = np.zeros((10, 1))
        label = []
        for parent, dirnames, filenames in os.walk(dirName):
            # print parent 
            # print dirnames 
            # print filenames 
            index = 0
            for dirname in dirnames:
                for subParent, subDirName, subFilenames in os.walk(parent + '/' + dirname):
                    for filename in subFilenames:
                        img = self.loadImg(subParent + '/' + filename)
                    tempImg = np.reshape(img, (-1, 1))
                    if index == 0:
                        dataMat = tempImg
                    else:
                        dataMat = np.column_stack((dataMat, tempImg))
                    label.append(subParent + '/' + filename)
                    index += 1
        return dataMat, label

    def PCA(self, dataMat, dimNum):
        ''''' 
        PCA函数，用于数据降维 
        :param dataMat: 样本矩阵 
        :param dimNum: 降维后的目标维度 
        :return: 降维后的样本矩阵和变换矩阵 
        '''

        # 均值化矩阵 
        meanMat = np.mat(np.mean(dataMat, 1)).T
        print '平均值矩阵维度', meanMat.shape
        diffMat = dataMat - meanMat
        # 求协方差矩阵，由于样本维度远远大于样本数目，所以不直接求协方差矩阵，采用下面的方法 
        # covMat = (diffMat.T * diffMat) / float(diffMat.shape[1])  # 归一化 
        covMat = np.cov(dataMat,bias=True) 
        # print '基本方法计算协方差矩阵为',covMat2 
        print '协方差矩阵维度', covMat.shape
        eigVals, eigVects = np.linalg.eig(np.mat(covMat))
        print '特征向量维度', eigVects.shape
        print '特征值', eigVals
        eigVects = diffMat * eigVects
        eigValInd = np.argsort(eigVals)
        eigValInd = eigValInd[::-1]
        eigValInd = eigValInd[:dimNum]  # 取出指定个数的前n大的特征值 
        print '选取的特征值', eigValInd
        eigVects = eigVects / np.linalg.norm(eigVects, axis=0)  # 归一化特征向量 
        redEigVects = eigVects[:, eigValInd]
        print '选取的特征向量', redEigVects.shape
        print '均值矩阵维度', diffMat.shape
        lowMat = redEigVects.T * diffMat
        print '低维矩阵维度', lowMat.shape
        return lowMat, redEigVects

    def compare(self, dataMat, testImg, label):
        ''''' 
        比较函数，这里只是用了最简单的欧氏距离比较，还可以使用KNN等方法，如需修改修改此处即可 
        :param dataMat: 样本矩阵 
        :param testImg: 测试图像矩阵，最原始形式 
        :param label: 标签矩阵 
        :return: 与测试图片最相近的图像文件名 
        '''
    
        testImg = cv2.resize(testImg, self.dsize)
        testImg = cv2.cvtColor(testImg, cv2.COLOR_RGB2GRAY)
        testImg = np.reshape(testImg, (-1, 1))
        lowMat, redVects = self.PCA(dataMat, self.dimNum)
        testImg = redVects.T * testImg
        print '检测样本变换后的维度', testImg.shape
        disList = []
        testVec = np.reshape(testImg, (1, -1))
        for sample in lowMat.T:
            disList.append(np.linalg.norm(testVec - sample))
        print disList
        sortIndex = np.argsort(disList)
        return label[sortIndex[0]]

    def predict(self, dirName, testFileName):
        ''''' 
        预测函数 
        :param dirName: 包含训练数据集的文件夹路径 
        :param testFileName: 测试图像文件名 
        :return: 预测结果 
        '''

        testImg = cv2.imread(testFileName)
        dataMat, label = self.createImgMat(dirName)
        print '加载图片标签', label
        ans = self.compare(dataMat, testImg, label)
        return ans


if __name__ == '__main__':
    eigenface = EigenFace(20, 50, (50, 50))
    print eigenface.predict('d:/face', 'D:/face_test/1.bmp') 
