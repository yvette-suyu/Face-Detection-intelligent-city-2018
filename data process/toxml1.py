Author: Nonove. nonove[at]msn[dot]com
# XML simple operation Examples and functions
# encoding = gbk

from xml.dom import minidom
import codecs


def write_xml_file(path, xmlDom, option = {'encoding':'utf-8'}):
    """ Generate xml file with writer
    params:
    string path xml file path
    Dom xmlDom xml dom
    dictionary option writer option {'indent': '', 'addindent':' ', 'newl':'\n', 'encoding':'utf-8'}
    returns:
    bool success return True else False
    """
    defaultOption = {'indent': '', 'addindent':' ', 'newl':'\n', 'encoding':'utf-8'}
    for k, v in defaultOption.iteritems():
        if k not in option:
            option[k] = v

    try:
        f=file(path, 'wb')
        writer = codecs.lookup(option['encoding'])[3](f)
        xmlDom.writexml(writer, encoding = option['encoding'], indent = option['indent'], \
        addindent = option['addindent'], newl = option['newl'])
        writer.close()
        return True
    except:
        print('Write xml file failed.... file:{0}'.format(path))
        return False



if __name__ == "__main__":
    # Create a xml dom
    xmlDom = minidom.Document()
    nonove = xmlDom.createElement('nonove')
    xmlDom.appendChild(nonove)

    # Generate a xml dom
    # Create child node, textnode, set attribute, appendChild
    for i in range(3):
        node = xmlDom.createElement('node')
        node.setAttribute('id', str(i))
        node.setAttribute('status', 'alive')
        textNode = xmlDom.createTextNode('node value ' + str(i))
        node.appendChild(textNode)
        nonove.appendChild(node)

    # Print xml dom
    # Print simple xml
    ## print(xmlDom.toxml())
    # Print pretty xml
    print('\n' + xmlDom.toprettyxml(indent=' '))

    # Save xml file with encoding utf-8
    option = {'indent': '', 'addindent':'', 'newl':'', 'encoding':'utf-8'}
    write_xml_file('nonove.xml', xmlDom, option)

    # Load xml dom from file
    xmlDom = minidom.parse('nonove.xml')
    # Get nonove node
    nonove = xmlDom.getElementsByTagName('nonove')[0]
    # Get node list
    nodes = xmlDom.getElementsByTagName('node')
    for node in nodes:
        # Get node attribute id
        nodeid = node.getAttribute('id')
        # Print node id and textnode value
        print('Node id: {0} textnode value: {1}'.format(nodeid, node.firstChild.nodeValue))

    for node in nodes:
        # Set attribute or remove attribute
        node.setAttribute('author', 'nonove')
        node.removeAttribute('status')

    # Remove node 1
    nonove.removeChild(nodes[1])

    print('\n' + xmlDom.toprettyxml(indent=' '))
