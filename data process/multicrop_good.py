import tensorflow as tf
import detect_util
import numpy as np
import os
import time
import cv2
import math
from PIL import Image,ImageStat

landmark=[]
www=[]
hhh=[]
bbbox=[]
class Detector:
    def __init__(self, model_path, gpu_fraction, device='/gpu:1',
                 min_size=20, thresh_list=[0.6, 0.7, 0.7], factor=0.709):
        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = gpu_fraction
        config.allow_soft_placement = True
        config.gpu_options.allow_growth = True
        self.gragh = tf.Graph()
        self.sess = tf.Session(config=config, graph=self.gragh)
        self.min_size = min_size
        self.thresh_list = thresh_list
        self.factor = factor
        self.device = device
        self.creat_gragh(model_path)


    def creat_gragh(self, model_path):
        print("begin to creat detector gragh")
        start = time.time()
        with self.gragh.as_default():
            with tf.device(self.device):
                with tf.variable_scope('pnet'):
                    data = tf.placeholder(tf.float32, (None, None, None, 3), 'input')
                    pnet = detect_util.PNet({'data': data})
                    pnet.load(os.path.join(model_path, 'det1.npy'), self.sess)
                with tf.variable_scope('rnet'):
                    data = tf.placeholder(tf.float32, (None, 24, 24, 3), 'input')
                    rnet = detect_util.RNet({'data': data})
                    rnet.load(os.path.join(model_path, 'det2.npy'), self.sess)
                with tf.variable_scope('onet'):
                    data = tf.placeholder(tf.float32, (None, 48, 48, 3), 'input')
                    onet = detect_util.ONet({'data': data})
                    onet.load(os.path.join(model_path, 'det3.npy'), self.sess)
        print("finish in {} s".format(time.time() - start))

    def detect_face(self, img, debug=False):
        """
        :param img:input image, load by opencv
        :return: result
        """
        img = np.array(img)
        factor_count = 0
        total_boxes = np.empty((0, 9))
        points = []
        h = img.shape[0]
        w = img.shape[1]
        minl = np.amin([h, w])
        m = 12.0 / self.min_size
        minl = minl * m
        # creat scale pyramid
        scales = []
        while minl >= 12:
            scales += [m * np.power(self.factor, factor_count)]
            minl = minl * self.factor
            factor_count += 1
        # first stage
        for j in range(len(scales)):
            scale = scales[j]
            hs = int(np.ceil(h * scale))
            ws = int(np.ceil(w * scale))
            im_data = detect_util.imresample(img, (hs, ws))
            im_data = (im_data - 127.5) * 0.0078125
            img_x = np.expand_dims(im_data, 0)
            img_y = np.transpose(img_x, (0, 2, 1, 3))
            out = self.sess.run(('pnet/conv4-2/BiasAdd:0', 'pnet/prob1:0'), feed_dict={'pnet/input:0': img_y})
            out0 = np.transpose(out[0], (0, 2, 1, 3))
            out1 = np.transpose(out[1], (0, 2, 1, 3))

            boxes, _ = detect_util.generateBoundingBox(out1[0, :, :, 1].copy(), out0[0, :, :, :].copy(), scale, self.thresh_list[0])

            # inter-scale nms
            pick = detect_util.nms(boxes.copy(), 0.5, 'Union')
            if boxes.size > 0 and pick.size > 0:
                boxes = boxes[pick, :]
                total_boxes = np.append(total_boxes, boxes, axis=0)

        numbox = total_boxes.shape[0]
        if numbox > 0:
            pick = detect_util.nms(total_boxes.copy(), 0.7, 'Union')
            total_boxes = total_boxes[pick, :]
            regw = total_boxes[:, 2] - total_boxes[:, 0]
            regh = total_boxes[:, 3] - total_boxes[:, 1]
            qq1 = total_boxes[:, 0] + total_boxes[:, 5] * regw
            qq2 = total_boxes[:, 1] + total_boxes[:, 6] * regh
            qq3 = total_boxes[:, 2] + total_boxes[:, 7] * regw
            qq4 = total_boxes[:, 3] + total_boxes[:, 8] * regh
            total_boxes = np.transpose(np.vstack([qq1, qq2, qq3, qq4, total_boxes[:, 4]]))
            total_boxes = detect_util.rerec(total_boxes.copy())
            total_boxes[:, 0:4] = np.fix(total_boxes[:, 0:4]).astype(np.int32)
            dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph = detect_util.pad(total_boxes.copy(), w, h)

        numbox = total_boxes.shape[0]
        if numbox > 0:
            # second stage
            tempimg = np.zeros((24, 24, 3, numbox))
            for k in range(0, numbox):
                tmp = np.zeros((int(tmph[k]), int(tmpw[k]), 3))
                tmp[dy[k] - 1:edy[k], dx[k] - 1:edx[k], :] = img[y[k] - 1:ey[k], x[k] - 1:ex[k], :]
                if tmp.shape[0] > 0 and tmp.shape[1] > 0 or tmp.shape[0] == 0 and tmp.shape[1] == 0:
                    tempimg[:, :, :, k] = detect_util.imresample(tmp, (24, 24))
                else:
                    return np.empty()
            tempimg = (tempimg - 127.5) * 0.0078125
            tempimg1 = np.transpose(tempimg, (3, 1, 0, 2))
            out = self.sess.run(('rnet/conv5-2/conv5-2:0', 'rnet/prob1:0'), feed_dict={'rnet/input:0': tempimg1})
            out0 = np.transpose(out[0])
            out1 = np.transpose(out[1])
            score = out1[1, :]
            ipass = np.where(score > self.thresh_list[1])
            total_boxes = np.hstack([total_boxes[ipass[0], 0:4].copy(), np.expand_dims(score[ipass].copy(), 1)])
            mv = out0[:, ipass[0]]
            if total_boxes.shape[0] > 0:
                pick = detect_util.nms(total_boxes, 0.7, 'Union')
                total_boxes = total_boxes[pick, :]
                total_boxes = detect_util.bbreg(total_boxes.copy(), np.transpose(mv[:, pick]))
                total_boxes = detect_util.rerec(total_boxes.copy())

        numbox = total_boxes.shape[0]
        if numbox > 0:
            # third stage
            total_boxes = np.fix(total_boxes).astype(np.int32)
            dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph = detect_util.pad(total_boxes.copy(), w, h)
            tempimg = np.zeros((48, 48, 3, numbox))
            for k in range(0, numbox):
                tmp = np.zeros((int(tmph[k]), int(tmpw[k]), 3))
                tmp[dy[k] - 1:edy[k], dx[k] - 1:edx[k], :] = img[y[k] - 1:ey[k], x[k] - 1:ex[k], :]
                if tmp.shape[0] > 0 and tmp.shape[1] > 0 or tmp.shape[0] == 0 and tmp.shape[1] == 0:
                    tempimg[:, :, :, k] = detect_util.imresample(tmp, (48, 48))
                else:
                    return np.empty()
            tempimg = (tempimg - 127.5) * 0.0078125
            tempimg1 = np.transpose(tempimg, (3, 1, 0, 2))
            out = self.sess.run(('onet/conv6-2/conv6-2:0', 'onet/conv6-3/conv6-3:0', 'onet/prob1:0'), feed_dict={'onet/input:0': tempimg1})
            out0 = np.transpose(out[0])
            out1 = np.transpose(out[1])
            out2 = np.transpose(out[2])
            score = out2[1, :]
            points = out1
            ipass = np.where(score > self.thresh_list[2])
            points = points[:, ipass[0]]
            total_boxes = np.hstack([total_boxes[ipass[0], 0:4].copy(), np.expand_dims(score[ipass].copy(), 1)])
            mv = out0[:, ipass[0]]

            w = total_boxes[:, 2] - total_boxes[:, 0] + 1
            h = total_boxes[:, 3] - total_boxes[:, 1] + 1
            points[0:5, :] = np.tile(w, (5, 1)) * points[0:5, :] + np.tile(total_boxes[:, 0], (5, 1)) - 1
            points[5:10, :] = np.tile(h, (5, 1)) * points[5:10, :] + np.tile(total_boxes[:, 1], (5, 1)) - 1
            if total_boxes.shape[0] > 0:
                total_boxes = detect_util.bbreg(total_boxes.copy(), np.transpose(mv))
                pick = detect_util.nms(total_boxes.copy(), 0.7, 'Min')
                total_boxes = total_boxes[pick, :]
                points = points[:, pick]

        result = []
        for bb, point in zip(total_boxes, np.transpose(points)):
            result_dic = {}
            result_dic['bbox'] = list(bb[:4])  # left_x, up_y, right_x, down_y
            bbbox.append(result_dic['bbox'])
            result_dic['width'] = bb[2] - bb[0]
            www.append(result_dic['width'])
            result_dic['height'] = bb[3] - bb[1]
            hhh.append(result_dic['height'])
            result_dic['area'] = (bb[2] - bb[0]) * (bb[3] - bb[1])
            result_dic['score'] = bb[4]
            result_dic['left_eye'] = [point[0], point[5]]
            landmark.append(result_dic['left_eye'])
            result_dic['right_eye'] = [point[1], point[6]]
            landmark.append(result_dic['right_eye'])
            result_dic['nose'] = [point[2], point[7]]
            landmark.append(result_dic['nose'])
            result_dic['left_mouth'] = [point[3], point[8]]
            landmark.append(result_dic['left_mouth'])
            result_dic['right_mouth'] = [point[4], point[9]]
            landmark.append(result_dic['right_mouth'])
            result.append(result_dic)
            #


            if debug:
                debug_img = img.copy()
                cv2.rectangle(debug_img, (int(bb[0]), int(bb[1])), (int(bb[2]), int(bb[3])), (0, 255, 0), 2)
                cv2.circle(debug_img, tuple(result_dic['left_eye']), 1, (255, 0, 0), 2)
                cv2.circle(debug_img, tuple(result_dic['right_eye']), 1, (255, 0, 0), 2)
                cv2.circle(debug_img, tuple(result_dic['nose']), 1, (255, 0, 0), 2)
                cv2.circle(debug_img, tuple(result_dic['left_mouth']), 1, (255, 0, 0), 2)
                cv2.circle(debug_img, tuple(result_dic['right_mouth']), 1, (255, 0, 0), 2)
                cv2.imwrite('/home/hszc/Desktop/ZHIHUICHENGSHI/version3/preprocess/2/debug000.png', debug_img)
        print('landmark:',landmark)
        print('result',result)
        return result


color_value=[]
imvar = 0
def colorvalue(im_path):
    """

    :param imagepath__: each image path,including part image
    :return:
    """
    imageData = Image.open(im_path)
    stat=ImageStat.Stat(imageData)
    imvar = stat.var
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
    color_value.append(imvar)
    # print(color_value)


def croppart(landmark,savefilepath):
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
    cv2.imwrite(savefilepath+'part1.png', part1)
    cv2.imwrite(savefilepath+'part2.png', part2)
    cv2.imwrite(savefilepath+'part3.png', part3)
    cv2.imwrite(savefilepath+'part4.png', part4)
    cv2.imwrite(savefilepath+'part5.png', part5)
    # newimg=im.copy()
    # cv2.imwrite(filepath+'xx.png',newimg)

    # cv2.imwrite('/home/hszc/Desktop/ZHIHUICHENGSHI/version3/preprocess/2/part5.png', part5)


if __name__ == '__main__':
    from glob import glob
    model_path = '/home/hszc/Desktop/ZHIHUICHENGSHI/version3/preprocess/model/mtcnn'
    dirroot = "/home/hszc/Desktop/ZHIHUICHENGSHI/version1/facenet/data/mydata/train_foruser"

    for dirpath, dirnames, filenames in os.walk(dirroot):

        print('dirpath', dirpath)
        print('dirname', dirnames)
        for filepath in filenames:
            im_path= os.path.join(dirpath, filepath)
            eachdirname = os.path.split(dirpath)[1]
            savefilepath='/home/hszc/Desktop/ZHIHUICHENGSHI/version3/preprocess/2/test/part/'+eachdirname+'/'+filepath+'/'
            detector = Detector(model_path, gpu_fraction=0.5)
            # im = cv2.imread(im_path)[:,:,::-1]
            im = cv2.imread(im_path)[:, :, :]
            results = detector.detect_face(im, debug=True)

            if not os.path.exists(savefilepath):
                os.makedirs(savefilepath)
            else:
                print('savepath is existed')
            print('crop to parts')
            try:
                with croppart(landmark,savefilepath):
                    orichara = []
                    color_value.append(colorvalue(im_path))
                    orichara.append(color_value)
                    color_value = []
                    print(orichara)

                    # for part images
                    rootdir = savefilepath
                    list = os.listdir(rootdir)
                    imgchara = []
                    for i in range(0, len(list)):

                        path = os.path.join(rootdir, list[i])
                        print(list[i])
                        if os.path.isfile(path):
                            color_value.append(colorvalue(path))
                        imgchara.append(color_value)
                        color_value = []
                    print('chara', imgchara)
            except:
                print('cannot detect face! next image!')
            # for original image



    # compare to detect occlusion



