basepath = '/home/hszc/Desktop/ZHIHUICHENGSHI/version1/facenet/data/mydata/'
csvfile = open(basepath+'test_a_gallery.csv', 'r')
ID_gallery =[]
ID =[]
for line in csvfile:
    ID_gallery.append(list(line.strip().split(',')))
for index in range(len(ID_gallery)):
    ID.append(ID_gallery[index][1])
# print(ID)
                with open('/home/hszc/Desktop/ZHIHUICHENGSHI/version1/facenet/runtest619/output/result2.txt','a') as f:
                    f.write('<?xml version="1.0" encoding="gb2312"?>'+'\n'+'<Message Version="1.0">'+'\n'
                            +'<Info  '                              ' evaluateType="1"'+'\n'+'mediaFile="dongnanmenwest_16_1920x1080_30" />'
                            +'\n'+'<Items>'+'\n')
                    f.close()

                for index in range(len(best_class_indices)):
                    here = str(best_class_indices[index])
                    if here in ID:
                         with open('/home/hszc/Desktop/ZHIHUICHENGSHI/version1/facenet/runtest619/output/result2.txt','a') as f:
                            f.write('<Item imageName="'+ str(namelist[index]) +'">' +'\n'+'<Label id = "'+ str(best_class_indices[index]) + '" />'+'\n'+'</Item>'+'\n')
                         f.close()
                    if here not in ID:
                         with open('/home/hszc/Desktop/ZHIHUICHENGSHI/version1/facenet/runtest619/output/result2.txt','a') as f:
                            f.write('<Item imageName="' + str(namelist[index]) + '">' + '\n' + '<Label id = "' + '0' + '" />' + '\n' + '</Item>' + '\n')
                         f.close()

                with open('/home/hszc/Desktop/ZHIHUICHENGSHI/version1/facenet/runtest619/output/result2.txt','a') as f:
                    f.write('</Items>'+'\n'+'</Message>'+'\n')
                f.close()
