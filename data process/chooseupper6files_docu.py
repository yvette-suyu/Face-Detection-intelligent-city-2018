import os
import cv2

dirpath='/home/hszc/Desktop/ZHIHUICHENGSHI/version1/facenet/data/mydata/train_new160/'

for parent, dirnames, filenames in os.walk(dirpath):
    filenamelist=filenames
    print(filenames)
    eachdirname = os.path.split(parent)[1]
    savefilepath = '/home/hszc/Desktop/ZHIHUICHENGSHI/version1/facenet/data/mydata/trainaaa/' + eachdirname + '/'
    if filenamelist:
        imgpath=parent
        if len(filenamelist)>6:
            for i in range(len(filenamelist)):

                print('Fname',filenamelist[i])
                img=cv2.imread(imgpath+'/'+filenamelist[i])
                if not os.path.exists(savefilepath):
                    os.makedirs(savefilepath)
                else:
                    print('savepath is existed')
                print('--ok--')

                cv2.imwrite(savefilepath+filenamelist[i],img)
        else:
            continue
