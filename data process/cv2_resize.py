import cv2
import os
# images_path="/media/hszc/data1/syh/zhijiang/ZJdata/DatasetA_train_20180813/train"
images_path='/media/hszc/data1/syh/zhijiang/ZJdata/DatasetA_test_20180813/DatasetA_test/test'
# save_path = '/media/hszc/data1/syh/zhijiang/ZJdata/DatasetA_train_20180813/train224/'
save_path = '/media/hszc/data1/syh/zhijiang/ZJdata/DatasetA_test_20180813/DatasetA_test/test224/'
for x, _, z in os.walk(images_path):
    # print(z)
    for pic in z:

        imgpath = os.path.join(images_path , pic)
        print(imgpath)
        img = cv2.imread(imgpath)
        a = img.shape
        resized = cv2.resize(img,(int(a[1]*3.5),int(a[0]*3.5)),interpolation=cv2.INTER_AREA)
        save = os.path.join(save_path,pic)
        cv2.imwrite(save,resized)
