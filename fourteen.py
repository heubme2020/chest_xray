import os
import pandas as pd
import shutil

def get_train_data():
    os.makedirs('fourteen', exist_ok=True)
    os.makedirs('fourteen/train', exist_ok=True)

    data = pd.read_csv('chest-xray-14/Data_Entry_2017.csv')
    data.rename(columns={'Image Index': 'image'}, inplace=True)
    data.rename(columns={'Finding Labels': 'classes'}, inplace=True)
    data = data.drop(columns=['Unnamed: 11'])
    data = data.drop(columns=['y]'])
    data = data.drop(columns=['OriginalImagePixelSpacing[x'])
    data = data.drop(columns=['Height]'])
    data = data.drop(columns=['OriginalImage[Width'])
    data = data.drop(columns=['View Position'])
    data = data.drop(columns=['Patient Gender'])
    data = data.drop(columns=['Patient Age'])
    data = data.drop(columns=['Patient ID'])
    data = data.drop(columns=['Follow-up #'])
    # data = data.dropna(subset=['Labels'])
    # # data.to_csv('detection/detection_label.csv', index=False)
    # print(data)
    # data = data.drop_duplicates(subset=['image', 'x', 'y', 'width'], keep='first')
    # data = data.reset_index(drop=True)
    # data['Labels'] = data['Finding Label']
    # data = data.drop(columns=['Finding Label'])
    print(data)

    columns = ['image', 'classes']
    dst_data = pd.DataFrame(columns=columns)
    for index, row in data.iterrows():
        basename = row['image']
        src_name = os.path.join('chest-xray-14/images', basename)
        dst_name = os.path.join('fourteen/train', basename)
        try:
            shutil.copy(src_name, dst_name)
            dst_data.loc[len(dst_data)] = row
        except:
            pass
    dst_data.to_csv('fourteen/append_train_label.csv', index=False)

def read_txt(txt_file):
    content = []
    try:
        with open(txt_file, 'r') as file:
            content = file.read()
            content = content.split()  # 默认按照空格进行分割
            # print(content)
            # print(content)
    except FileNotFoundError:
        print("File not found.")
    except IOError:
        print("Error reading the file.")
    return content

def get_txt_label_data():
    folder = 'Shenzhen-Hospital-CXR-Set/ClinicalReadings'
    txt_files = [os.path.abspath(os.path.join(folder, file)) for file in os.listdir(folder) if
                 file.lower().endswith('.txt')]
    columns = ['image', 'classes']
    dst_data = pd.DataFrame(columns=columns)
    for i in range(len(txt_files)):
        txt_file = txt_files[i]
        content = read_txt(txt_file)
        print(content)
        base_name = os.path.basename(txt_file)
        base_name = base_name.replace('.txt', '.png')
        src_name = os.path.join('Shenzhen-Hospital-CXR-Set/CXR_png', base_name)
        dst_name = os.path.join('fourteen/train', base_name)
        try:
            Labels = content[2:]
            Labels = "_".join(Labels)
            row = {'image': base_name, 'classes': Labels}
            shutil.copy(src_name, dst_name)
            dst_data.loc[len(dst_data)] = row
        except:
            pass
    dst_data.to_csv('fourteen/append_train_label.csv', index=False)


if __name__ == '__main__':
    # get_train_data()
    dataA = pd.read_csv('fourteen/append_train_label.csv')
    dataB = pd.read_csv('fourteen/train_label.csv')
    data = pd.concat([dataA, dataB], ignore_index=True)
    data = data.sample(frac=1, random_state=1024)
    data = data.reset_index(drop=True)
    print(data)
    data.to_csv('fourteen/train_label.csv', index=False)
    # get_train_data()
    # get_txt_label_data()