import os
import cv2

dirpath='/home/hszc/Desktop/ZHIHUICHENGSHI/jiayou/data/raw/'

for parent, dirnames, filenames in os.walk(dirpath):
    filenamelist=filenames
    print(filenames)
    eachdirname = os.path.split(parent)[1]
    savefilepath = '/home/hszc/Desktop/ZHIHUICHENGSHI/jiayou/data/use/' + eachdirname + '/'
    if filenamelist:
        imgpath=parent
        for i in range(5):

            print('Fname',filenamelist[i])
            img=cv2.imread(imgpath+'/'+filenamelist[i])
            if not os.path.exists(savefilepath):
                os.makedirs(savefilepath)
            else:
                print('savepath is existed')
            print('--ok--')

            cv2.imwrite(savefilepath+filenamelist[i],img)

