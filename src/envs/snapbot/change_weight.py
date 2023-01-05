import numpy as np
import xml.etree.ElementTree as ET
import os
def change_xml():
        for i in range(100):
                xml_path = os.path.join(os.path.dirname(__file__), 'xml/snapbot_4/robot_4_1245_{}.xml'.format(i))
                target_xml = open(xml_path, 'rt', encoding='UTF8')
                tree = ET.parse(target_xml)
                root = tree.getroot()
                tag=root.find('worldbody').find('include')
                tag.attrib["file"] = "snapbot_4_1245_{}.xml".format(i)
                tree.write(xml_path)

if __name__ == "__main__":
    change_xml()