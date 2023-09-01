import xml.etree.ElementTree as ET
import os

def view_xml(folder):
    xml_files = [os.path.abspath(os.path.join(folder, file)) for file in os.listdir(folder) if
                 file.lower().endswith('.xml')]
    for i in range(len(xml_files)):
        xml_file = xml_files[i]
        # 使用ElementTree的parse()函数打开XML文件
        tree = ET.parse(xml_file)
        # 获取XML文档的根元素
        root = tree.getroot()
        # 遍历根元素和它的子元素，并打印它们的标签名和文本内容
        # for element in root:
        #     print(f"Element: {element.tag}, Text: {element.text}")
        # print("Root element:", root.tag)
        xml_content = ET.tostring(root, encoding='utf-8')
        print(xml_content.decode('utf-8'))
if __name__ == '__main__':
    view_xml('ecgen-radiology')

# # 指定XML文件路径
# xml_file_path = "path/to/your/xmlfile.xml"
#
# # 使用ElementTree的parse()函数打开XML文件
# tree = ET.parse(xml_file_path)
#
# # 获取XML文档的根元素
# root = tree.getroot()
#
# # 现在你可以对XML文档进行解析和操作，比如访问元素、获取属性、遍历子元素等
# # 例如，输出根元素的标签名
# print("Root element:", root.tag)