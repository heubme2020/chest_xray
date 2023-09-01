import os
import pandas as pd
from PIL import Image
import numpy as np
import shutil


def is_subset(subset_str, target_str):
    subset_set = set(subset_str)
    target_set = set(target_str)

    return subset_set.issubset(target_set)

if __name__ == '__main__':
    folder = 'Shenzhen-Hospital-CXR-Set/CXR_png'
    mask_folder = 'Shenzhen-Hospital-CXR-Set/masks'
    png_files = [os.path.abspath(os.path.join(folder, file)) for file in os.listdir(folder) if
                 file.lower().endswith('.png')]
    mask_png_files = [os.path.abspath(os.path.join(mask_folder, file)) for file in os.listdir(mask_folder) if
                 file.lower().endswith('.png')]
    for i in range(len(png_files)):
        png_file = png_files[i]
        base_name = os.path.basename(png_file)
        mask_files = []
        for j in range(len(mask_png_files)):
            mask_file = mask_png_files[j]
            mask_basename = os.path.basename(mask_file)
            mask_basename_splits = mask_basename.split('_')
            mask_basename_splits_selected = [mask_basename_splits[0], mask_basename_splits[1], mask_basename_splits[-1]]
            mask_basename = "_".join(mask_basename_splits_selected)
            if base_name == mask_basename:
                mask_files.append(mask_file)
        image = Image.open(png_file)
        image_width, image_height = image.size
        mask_array = np.zeros((image_height, image_width, 1), dtype=np.uint8)
        if len(mask_files) == 0:
            continue
        for k in range(len(mask_files)):
            mask_array = np.squeeze(mask_array)
            mask_file = mask_files[k]
            mask_image = Image.open(mask_file)
            mask_image = mask_image.resize((image_width, image_height))
            mask_array_add = np.array(mask_image)
            mask_array = mask_array + mask_array_add
            mask_array = np.clip(mask_array, 0, 255)
        mask_array = Image.fromarray(mask_array.astype(np.uint8))
        # print(mask_array)

        src_image_name = png_file
        dst_image_name = os.path.join('segment/src_image', base_name)
        shutil.copy(src_image_name, dst_image_name)
        dst_mask_name = os.path.join('segment/mask_image', base_name)
        mask_array.save(dst_mask_name)

        # left_file = os.path.join('MontgomerySet/ManualMask/leftMask', base_name)
        # right_file = os.path.join('MontgomerySet/ManualMask/rightMask', base_name)
        # file_name = os.path.join('MontgomerySet/ManualMask', base_name)
        # mask_image_left = Image.open(left_file)
        # mask_image_right = Image.open(right_file)
        #
        # # 将图像转换为NumPy数组
        # mask_array1 = np.array(mask_image_left)*255
        # mask_array2 = np.array(mask_image_right)*255
        #
        # # 将两个数组相加
        # merged_array = mask_array1 + mask_array2
        #
        # # 限制数组值在0到255之间
        # merged_array = np.clip(merged_array, 0, 255)
        #
        # # 将数组转换回图像
        # merged_image = Image.fromarray(merged_array.astype(np.uint8))
        #
        # # 保存合并结果
        # merged_image.save(file_name)
