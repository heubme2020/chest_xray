import pandas as pd
from PIL import Image
import numpy as np
import glob
import os
from tqdm import tqdm
import pydicom

def rle_to_mask():
    os.makedirs('src_image', exist_ok=True)
    os.makedirs('mask_image', exist_ok=True)
    label_data = pd.read_csv('SIIM_TRAIN_TEST/train-rle.csv')
    # print(label_data)
    folder = 'SIIM_TRAIN_TEST/dicom-images-train'
    dcm_files = glob.glob(folder + '/**/*.dcm', recursive=True)
    files_num = len(dcm_files)
    for i in tqdm(range(files_num)):
        try:
            dcm_file = dcm_files[i]
            ds = pydicom.dcmread(dcm_file)
            # 获取图像数据
            image_data = ds.pixel_array
            # 将像素数据转换为 PIL Image 对象
            image = Image.fromarray(image_data)
            base_name = os.path.basename(dcm_file)
            png_base_name = base_name.replace('dcm', 'png')
            png_name = 'src_image/' + png_base_name

            width = image.width
            height = image.height
            mask = np.zeros(width * height)

            ImageId = base_name.replace('.dcm', '')
            filtered_rows = label_data[label_data['ImageId'] == ImageId]
            # print(filtered_rows)
            rle_values = filtered_rows[' EncodedPixels']
            is_empty = rle_values.empty
            if is_empty:
                continue
            else:
                rle = rle_values.tolist()[0]
                if rle != '-1':
                    rle_values = rle.split(' ')
                    array = np.asarray([int(x) for x in rle_values])
                    starts = array[0::2]
                    lengths = array[1::2]
                    current_position = 0
                    for index, start in enumerate(starts):
                        current_position += start
                        mask[current_position:current_position + lengths[index]] = 255
                        current_position += lengths[index]
                mask = mask.reshape(width, height)
                mask = mask.astype(np.uint8)
                mask_image = Image.fromarray(mask)  # 转为PIL图像
                mask_name = 'mask_image/' + png_base_name
                mask_image.save(mask_name)
                image.save(png_name)
        except:
            pass




if __name__ == '__main__':
    rle_to_mask()





