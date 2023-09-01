import pydicom
import os
import pandas as pd
from tqdm import tqdm
from PIL import Image
import glob

def read_dcms(folder):
    dcm_files = [os.path.abspath(os.path.join(folder, file)) for file in os.listdir(folder) if
                 file.lower().endswith('.dcm')]
    for i in range(len(dcm_files)):
        dcm_file = dcm_files[i]
        ds = pydicom.dcmread(dcm_file)
        print(ds)
        image_data = ds.pixel_array
        print(image_data)

def get_png_data(folder):
    os.makedirs('png', exist_ok=True)
    # dcm_files = [os.path.abspath(os.path.join(folder, file)) for file in os.listdir(folder) if
    #              file.lower().endswith('.dcm')]
    dcm_files = glob.glob(folder + '/**/*.dcm', recursive=True)
    label_data = pd.DataFrame(columns=['PatientID', 'PngBasename', 'PatientSex', 'PatientAge', 'ViewPosition'])
    for i in tqdm(range(len(dcm_files))):
        try:
            dcm_file = dcm_files[i]
            png_file = dcm_file.replace('dcm', 'png')
            png_basename = os.path.basename(png_file)
            png_name = 'png/' + png_basename
            ds = pydicom.dcmread(dcm_file)
            age_value = ds.PatientAge
            sex_value = ds.PatientSex
            view_position = ds.ViewPosition
            patient_id = ds.PatientID
            row_data = [patient_id, png_basename, sex_value, age_value, view_position]
            print(row_data)
            label_data.loc[len(label_data)] = row_data

            # 获取图像数据
            image_data = ds.pixel_array
            # 将像素数据转换为 PIL Image 对象
            image = Image.fromarray(image_data)
            image.save(png_name)
        except:
            pass
    print(label_data)
    label_data.to_csv('label.csv', index=False)

        # print(age_value)
        # print(sex_value)
        # print(view_position)
        # print(ds)
        # print(ds)
        # image_data = ds.pixel_array
        # print(image_data)
def merge_data(folder):
    label_csv_file = os.path.join(folder, 'label.csv')
    train_label_csv_file = os.path.join(folder, 'train_label.csv')
    stage_2_train_label_csv_file = os.path.join(folder, 'stage_2_train_labels.csv')
    stage_2_detailed_class_info_csv_file = os.path.join(folder, 'stage_2_detailed_class_info.csv')
    label_data = pd.read_csv(label_csv_file)
    train_label_data = pd.read_csv(train_label_csv_file)
    stage_2_train_label_data = pd.read_csv(stage_2_train_label_csv_file)
    stage_2_train_label_data.rename(columns={'patientId': 'PatientID'}, inplace=True)
    stage_2_detailed_class_info_data = pd.read_csv(stage_2_detailed_class_info_csv_file)
    stage_2_detailed_class_info_data.rename(columns={'patientId': 'PatientID'}, inplace=True)
    label_data = pd.concat([train_label_data, label_data], ignore_index=True)
    label_data = label_data.reset_index(drop=True)
    print(label_data)
    print(stage_2_train_label_data)
    label_data = pd.merge(label_data, stage_2_train_label_data, on='PatientID', how='outer')
    print(label_data)
    label_data.dropna(subset=['PatientID'], inplace=True)
    label_data = label_data.reset_index(drop=True)
    print(label_data)
    label_data = pd.merge(label_data, stage_2_detailed_class_info_data, on='PatientID', how='outer')
    print(label_data)
    label_data.dropna(subset=['PatientID'], inplace=True)
    label_data = label_data.reset_index(drop=True)
    print(label_data)
def merge_chest_xray(folder):
    BBox_file = os.path.join(folder, 'BBox_List_2017.csv')
    BBox_data = pd.read_csv(BBox_file)
    BBox_data.rename(columns={'Image Index': 'PngBasename'}, inplace=True)
    BBox_data.rename(columns={'w': 'width'}, inplace=True)
    BBox_data.rename(columns={'h': 'height'}, inplace=True)
    BBox_data.insert(0, 'PatientID', BBox_data['PngBasename'])
    BBox_data['PatientID'] = BBox_data['PatientID'].apply(lambda x:x.replace('.png', ''))
    print(BBox_data.columns)
    print(BBox_data)
    BBox_data.to_csv('BBox_data.csv', index=False)


if __name__ == '__main__':
    get_png_data('SIIM_TRAIN_TEST')
