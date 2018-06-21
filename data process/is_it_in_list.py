import csv

basepath = '/home/hszc/Desktop/ZHIHUICHENGSHI/version1/facenet/data/mydata/'
csvfile = open(basepath+'test_a_gallery.csv', 'r')
ID_gallery =[]
ID =[]
for line in csvfile:
    ID_gallery.append(list(line.strip().split(',')))
for index in range(len(ID_gallery)):
    ID.append(ID_gallery[index][1])
print(ID)

index = [2,4,6,8,9,22,44,65,76,89]
for here in index:
    print('hi!',here)
    here = str(here)
    if here in ID:
        print('6666666666666!')
    if here not in ID:
        print('sad!!!')
