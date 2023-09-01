import os
import pandas as pd
import shutil
from PIL import Image, ImageDraw
import keyboard
import pyautogui

def get_train_data():
    os.makedirs('detection', exist_ok=True)
    os.makedirs('detection/train', exist_ok=True)
    os.makedirs('detection/train_append', exist_ok=True)

    data = pd.read_csv('Shenzhen-Hospital-CXR-Set/shenzhen_consensus_roi.csv')
    print(data)
    data.rename(columns={'patientId': 'image'}, inplace=True)
    # data['annotation'] = data['annotation'].fillna(1)
    data.rename(columns={'x_dis': 'x'}, inplace=True)
    data.rename(columns={'y_dis': 'y'}, inplace=True)
    data.rename(columns={'width_dis': 'width'}, inplace=True)
    data.rename(columns={'height_dis': 'height'}, inplace=True)
    # data = data.drop(columns=['PatientID'])
    # # data = data.drop(columns=['PatientSex'])
    # # data = data.drop(columns=['PatientAge'])
    # # data = data.drop(columns=['ViewPosition'])
    # # data = data.drop(columns=['Target'])
    # # data = data.dropna(subset=['Labels'])
    # # # data.to_csv('detection/detection_label.csv', index=False)
    # # print(data)
    # # data = data.drop_duplicates(subset=['image', 'x', 'y', 'width'], keep='first')
    # # data = data.reset_index(drop=True)
    # data['Labels'] = data['Finding Label']
    # data = data.drop(columns=['Finding Label'])
    # print(data)
    # #
    columns = ['image', 'x', 'y', 'width', 'height', 'Labels']
    dst_data = pd.DataFrame(columns=columns)
    for index, row in data.iterrows():
    #     if row['annotation'] == 1:
    #         image_name = row['image'].replace('jpg', 'png')
    #         row_add = {'image': image_name, 'x':0, 'y':0, 'width':0, 'height':0, 'Labels':''}
    #         src_name = os.path.join('object-CXR/train', row['image'])
    #         dst_name = os.path.join('detection/train_append', image_name)
    #         try:
    #             jpeg_image = Image.open(src_name)
    #             jpeg_image.save(dst_name, format='PNG')
    #             dst_data.loc[len(dst_data)] = row_add
    #         except:
    #             pass
    #     else:
    #         annotation_list = row['annotation'].split(' ')
    #         annotation_list.pop(0)
    #         box_num = int(len(annotation_list)/4)
    #         if box_num > 11:
    #             continue
    #         for i in range(box_num):
    #             image_name = row['image'].replace('jpg', 'png')
    #             x = int(annotation_list[i*4])
    #             y = int(annotation_list[i*4 + 1])
    #             width = int(annotation_list[i*4 + 2])
    #             width = abs(width - x)
    #             height = annotation_list[i*4 + 3]
    #             height = height.split(';')[0]
    #             height = int(height)
    #             height = abs(height - y)
    #             row_add = {'image': image_name, 'x':x, 'y':y, 'width':width, 'height':height, 'Labels':''}
    #             print(row_add)
    #             src_name = os.path.join('object-CXR/train', row['image'])
    #             dst_name = os.path.join('detection/train_append', image_name)
    #             try:
    #                 jpeg_image = Image.open(src_name)
    #                 if os.path.exists(dst_name):
    #                     pass
    #                 else:
    #                     jpeg_image.save(dst_name, format='PNG')
    #                 dst_data.loc[len(dst_data)] = row_add
    #                 # # 创建Draw对象以在图像上绘制矩形
    #                 # draw = ImageDraw.Draw(jpeg_image)
    #                 # box_coordinates = (x, y, x+width, y+height)
    #                 # draw.rectangle(box_coordinates, outline="red", width=3)  # 使用红色绘制矩形框
    #                 # 显示图像
    #                 # jpeg_image.show()
    #                 # keyboard.wait("space")
    #                 # pyautogui.hotkey('ctrl', 'w')
    #             except:
    #                 pass
        basename = row['image']
        src_name = os.path.join('Shenzhen-Hospital-CXR-Set/CXR_png', basename)
    #     image = Image.open(src_name)
    #     box_coordinates = (100, 100, 300, 300)  # (x1, y1, x2, y2)
    #
    #     # 创建Draw对象以在图像上绘制矩形
    #     draw = ImageDraw.Draw(image)
    #     draw.rectangle(box_coordinates, outline="red", width=3)  # 使用红色绘制矩形框
        dst_name = os.path.join('detection/train', basename)
        try:
            shutil.copy(src_name, dst_name)
            dst_data.loc[len(dst_data)] = row
        except:
            pass
    dst_data.to_csv('detection/append_detection_label.csv', index=False)


if __name__ == '__main__':
    # get_train_data()
    dataA = pd.read_csv('detection/append_detection_label.csv')
    dataB = pd.read_csv('detection/detection_label.csv')
    data = pd.concat([dataA, dataB], ignore_index=True)
    data = data.sample(frac=1, random_state=1024)
    data = data.reset_index(drop=True)
    print(data)
    data.to_csv('detection/detection_label.csv', index=False)
    # get_train_data()