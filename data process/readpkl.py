'''
@ author yvette_suyu
2018.6.20

'''


import pickle
#check the version of python when you get the .pkl file.
# If it was python3 , you need to change the interpreter config into python3.6
# If it was python2 , use the python2.7

f = open('/home/hszc/Desktop/ZHIHUICHENGSHI/version1/facenet/models/facenet/testmodel/lfw_classifier.pkl','rb')
name,cls = pickle.load(f)
f.close()
print ('name:',name,'cls:',cls)   #show file
