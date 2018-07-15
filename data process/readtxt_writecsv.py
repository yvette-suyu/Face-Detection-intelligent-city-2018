import csv

file1 = open('/home/hszc/Desktop/ZHIHUICHENGSHI/jiayou/datause/result/result.txt', 'r')
result1 = []

for c in file1.readlines():
    c_array = c.split(",")
    result1.append(c_array[0])

# print(result1)



file2 = open('/home/hszc/Desktop/ZHIHUICHENGSHI/jiayou/datause/result/label.txt', 'r')
result2 = []
label2 = []
for c in file2.readlines():
    c_array = c.split(" ")
    result2.append(c_array)

# print(result)
for e in range(0,1):
    label2.append(result2[e][4][0:-1])

for a in range(1,10):
    label2.append(result2[a][5][0:-1])

for b in range(10,100):
    label2.append(result2[b][4][0:-1])
for c in range(100,1000):
    label2.append(result2[c][3][0:-1])
for d in range(1000,len(result2)-1):
    label2.append(result2[d][2][0:-1])

# print(zip(result1,label2))
# print(label2)
# for i in range(len(result1)-1):
#     print(result1[i],label2[i])


out = open('/home/hszc/Desktop/ZHIHUICHENGSHI/jiayou/datause/result/result.csv', 'a')
# out = open('/home/hszc/Desktop/ZHIHUICHENGSHI/version1/facenet/runtest619/test/resultfor/result.csv','a', newline='')
csv_write = csv.writer(out, dialect='excel')
for index in range(1292):
    newlist = [result1[index], label2[index]]
    csv_write.writerow(newlist)
