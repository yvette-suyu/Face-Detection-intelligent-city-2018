if __name__ == '__main__':
    from glob import glob
    import fnmatch
    model_path = '/home/hszc/Desktop/ZHIHUICHENGSHI/version3/preprocess/model/mtcnn'
    dirroot = "/home/hszc/Desktop/ZHIHUICHENGSHI/version1/facenet/data/mydata/train_foruser"

    for dirpath, dirnames, filenames in os.walk(dirroot):
        # # dirpath.sort()
        # dirnames=fnmatch.filter(dirnames,'*')
        # filenames=fnmatch.filter(filenames,'*.png')
        # dirnames=sorted(dirnames)
        # filenames=sorted(filenames)
        print('dirpath', dirpath)
        print('dirname', dirnames)
        for filepath in filenames:
            im_path= os.path.join(dirpath, filepath)
            eachdirname = os.path.split(dirpath)[1]
            # im_path = '/home/hszc/Desktop/ZHIHUICHENGSHI/version3/preprocess/2/test/debug_crop/0.png'
            savefilepath='/home/hszc/Desktop/ZHIHUICHENGSHI/version3/preprocess/2/test/part/'+eachdirname+'/'+filepath+'/'
            # orifilepath='/home/hszc/Desktop/ZHIHUICHENGSHI/version3/preprocess/2/test/'+eachdirname
            # if not os.path.exists(orifilepath):
            #     os.makedirs(orifilepath)
            # else:
            #     print('orifiledir is existed')
            detector = Detector(model_path, gpu_fraction=0.5)
            # im = cv2.imread(im_path)[:,:,::-1]
            im = cv2.imread(im_path)[:, :, :]
            results = detector.detect_face(im, debug=True)

            if not os.path.exists(savefilepath):
                os.makedirs(savefilepath)
            else:
                print('savepath is existed')
            print('crop to parts')
            croppart(landmark,savefilepath)

            # for original image

            orichara=[]
            color_value.append(colorvalue(im_path))
            orichara.append(color_value)
            color_value=[]
            print(orichara)

            # for part images
            rootdir = savefilepath
            list = os.listdir(rootdir)
            imgchara=[]
            for i in range(0, len(list)):

                path = os.path.join(rootdir, list[i])
                print(list[i])
                if os.path.isfile(path):
                    color_value.append(colorvalue(path))
                imgchara.append(color_value)
                color_value = []
            print('chara',imgchara)

    # compare to detect occlusion
