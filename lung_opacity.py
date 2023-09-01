import os
import pandas as pd
import shutil

def get_train_data():
    os.makedirs('lung_opacity', exist_ok=True)
    os.makedirs('lung_opacity/train', exist_ok=True)
    os.makedirs('lung_opacity/train/normal', exist_ok=True)
    os.makedirs('lung_opacity/train/opacity', exist_ok=True)
    os.makedirs('lung_opacity/train/else', exist_ok=True)

    data = pd.read_csv('rsna-pneumonia-detection-challenge/label_data.csv')
    data = data.dropna(subset=['class'])
    for index, row in data.iterrows():
        basename = row['PngBasename']
        src_name = os.path.join('rsna-pneumonia-detection-challenge/train', basename)
        classname = row['class']
        if classname == 'Normal':
            dst_name = os.path.join('lung_opacity/train/normal', basename)
            try:
                shutil.copy(src_name, dst_name)
            except:
                pass
        elif classname == 'Lung Opacity':
            dst_name = os.path.join('lung_opacity/train/opacity', basename)
            try:
                shutil.copy(src_name, dst_name)
            except:
                pass
        elif classname == 'No Lung Opacity / Not Normal':
            dst_name = os.path.join('lung_opacity/train/else', basename)
            try:
                shutil.copy(src_name, dst_name)
            except:
                pass


if __name__ == '__main__':
    get_train_data()




