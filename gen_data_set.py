import os
import shutil
from  tqdm import tqdm
import pandas as pd

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

def gen_gender_data():
    os.makedirs('gender', exist_ok=True)
    os.makedirs('gender/train', exist_ok=True)
    os.makedirs('gender/train/female', exist_ok=True)
    os.makedirs('gender/train/male', exist_ok=True)
    folder = 'MontgomerySet/CXR_png'
    png_files = [os.path.abspath(os.path.join(folder, file)) for file in os.listdir(folder) if
                 file.lower().endswith('.png')]
    folder_txt = 'MontgomerySet/ClinicalReadings'
    for i in tqdm(range(len(png_files))):
        png_file = png_files[i]
        png_basename = os.path.basename(png_file)
        txt_basename = png_basename.replace('png', 'txt')
        txt_file = os.path.join(folder_txt, txt_basename)
        content = read_txt(txt_file)
        print(content)
        if len(content) != 0:
            gender = content[2].lower()
            if gender == 'f':
                png_dst_file = 'gender/train/female/' + png_basename
                if os.path.exists(png_dst_file):
                    continue
                else:
                    shutil.copy(png_file, png_dst_file)
            elif gender == 'm':
                png_dst_file = 'gender/train/male/' + png_basename
                if os.path.exists(png_dst_file):
                    continue
                else:
                    shutil.copy(png_file, png_dst_file)
    # print(png_files)
def gen_test_data(folder):
    test_folder = 'gender/test_append'
    os.makedirs(test_folder, exist_ok=True)
    test_female_folder = 'gender/test_append/female'
    os.makedirs(test_female_folder, exist_ok=True)
    test_male_folder = 'gender/test_append/male'
    os.makedirs(test_male_folder, exist_ok=True)

    image_folder = folder + '/images'
    csv_file = folder + '/shenzhen_metadata.csv'
    label_data = pd.read_csv(csv_file, engine='pyarrow')
    for index, row in label_data.iterrows():
        image_src_name = os.path.join(image_folder, row['study_id'])
        if row['sex'].lower() == 'female':
            image_dst_name = os.path.join(test_female_folder, row['study_id'])
            shutil.copy(image_src_name, image_dst_name)
        elif row['sex'].lower() == 'male':
            image_dst_name = os.path.join(test_male_folder, row['study_id'])
            shutil.copy(image_src_name, image_dst_name)

def gen_projection_data():
    projection_folder = 'projection'
    os.makedirs(projection_folder, exist_ok=True)
    projection_train_folder = 'projection/train'
    os.makedirs(projection_train_folder, exist_ok=True)
    projection_frontal_folder = 'projection/train/frontal'
    os.makedirs(projection_frontal_folder, exist_ok=True)
    projection_lateral_folder = 'projection/train/lateral'
    os.makedirs(projection_lateral_folder, exist_ok=True)

    image_folder = 'chest-xrays-indiana-university/images'
    csv_file = 'chest-xrays-indiana-university/indiana_projections.csv'
    label_data = pd.read_csv(csv_file, engine='pyarrow')
    for index, row in label_data.iterrows():
        image_src_name = os.path.join(image_folder, row['filename'])
        if row['projection'].lower() == 'frontal':
            image_dst_name = os.path.join(projection_frontal_folder, row['filename'])
            shutil.copy(image_src_name, image_dst_name)
        elif row['projection'].lower() == 'lateral':
            image_dst_name = os.path.join(projection_lateral_folder, row['filename'])
            shutil.copy(image_src_name, image_dst_name)

if __name__ == '__main__':
    # model = Network()
    # print(model.eval())
    # Define the loss function with Classification Cross-Entropy loss and an optimizer with Adam optimizer
    # loss_fn = nn.CrossEntropyLoss()
    # optimizer = Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
    # gen_gender_data()
    gen_projection_data()