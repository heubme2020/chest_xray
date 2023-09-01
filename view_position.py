import pandas as pd
import os
import shutil

if __name__ == '__main__':
    view_data = pd.read_csv('rsna-pneumonia-detection-challenge/label_data.csv')
    view_data = view_data[['PngBasename', 'ViewPosition']]
    view_data = view_data.drop_duplicates()
    view_data = view_data.reset_index(drop=True)
    os.makedirs('view', exist_ok=True)
    os.makedirs('view/train', exist_ok=True)
    os.makedirs('view/train/PA', exist_ok=True)
    os.makedirs('view/train/AP', exist_ok=True)
    folder = 'rsna-pneumonia-detection-challenge/train'
    for index, row in view_data.iterrows():
        try:
            if row['ViewPosition'] == 'PA':
                src_name = os.path.join(folder, row['PngBasename'])
                dst_name = os.path.join('view/train/PA', row['PngBasename'])
                shutil.copy(src_name, dst_name)
            elif row['ViewPosition'] == 'AP':
                src_name = os.path.join(folder, row['PngBasename'])
                dst_name = os.path.join('view/train/AP', row['PngBasename'])
                shutil.copy(src_name, dst_name)
        except:
            pass