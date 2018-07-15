def get_image_paths_and_labels(dataset):

import os
# imagename = []
#rootdir = '/home/hszc/Desktop/ZHIHUICHENGSHI/version1/facenet/data/mydata/test_new160'
rootdir = dataset
#solution 1
# list = os.listdir(rootdir)
# print('content',list)
#
# for i in range(0,len(list)):
#     path = os.path.join(rootdir,list[i])
#     if os.path.isdir(path):
#         print('Im dir')
#         imagepath = os.
#         if os.path.isfile(path):
#             imagename.append(os.listdir(path))
#             print(imagename)

#solution 2
# for root,dirs,files in (os.walk(rootdir)):
#     print(root)
#     print(dirs)
#     print(files)

#solution 3
print('path is !!!!!!!', paths)
# imagename_ = os.path.basename(paths)
imagename_ = str(paths).split("/")[-1]
imagename.append(imagename_)
print('!!!!!!!!!!!!!!!!!imagenameis', imagename)
