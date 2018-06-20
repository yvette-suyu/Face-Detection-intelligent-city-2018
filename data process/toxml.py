

#xmlTest_write.py
# -*- coding: utf-8 -*-

import xml.dom.minidom

def GenerateXml():
    impl = xml.dom.minidom.getDOMImplementation()

    dom = impl.createDocument(None, 'emps', None)
    root = dom.documentElement
    employee = dom.createElement('emp')

    #
    employee.setAttribute("empno","1111")
    root.appendChild(employee)

    #
    #ename
    nameE=dom.createElement('ename')
    nameT=dom.createTextNode('jack')
    nameE.appendChild(nameT)
    #
    nameE.setAttribute("lastname","k")

    employee.appendChild(nameE)
    #age
    nameE=dom.createElement('age')
    nameT=dom.createTextNode('33')
    nameE.appendChild(nameT)

    employee.appendChild(nameE)

    f= open('/home/hszc/Desktop/ZHIHUICHENGSHI/version1/facenet/runtest619/output/result.xml', 'w') #change w to a append
    dom.writexml(f, addindent=' ', newl='\n')
    f.close()

GenerateXml()
