import numpy as np
import cv2

BLACK = [0, 0, 0]

img = cv2.imread('/home/hszc/Desktop/ZHIHUICHENGSHI/version1/facenet/data/mydata/train_foruser/0/6834b12186cb4e21bca59c9fcab9fa9f.png')

constant = cv2.copyMakeBorder(img, 0, 2, 0, 2, cv2.BORDER_CONSTANT, value=BLACK)
# copyMakeBorder(src, top, bottom, left, right, borderType[, dst[, value]])

cv2.imwrite('/home/hszc/Desktop/ZHIHUICHENGSHI/version3/preprocess/2/test/test/padding.png', constant)
