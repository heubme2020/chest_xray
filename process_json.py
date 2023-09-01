import os
import json


def find_json_files(folder_path):
    json_files = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".json"):
                json_files.append(os.path.join(root, file))
    return json_files


def process_json_file(file_path):
    with open(file_path, "r", encoding="utf-8") as json_file:
        data = json.load(json_file)

    return data

def get_train_data():
    folder_path = 'Shenzhen-Hospital-CXR-Set/Annotations/Annotations_json/SeparateFiles'  # 替换为实际的文件夹路径
    json_files = find_json_files(folder_path)
    print(len(json_files))

    for json_file in json_files:
        data = process_json_file(json_file)
        keys = list(data.keys())
        value = data[keys[0]]
        # print(value)
        image_name = value['filename']
        regions = value['regions']
        print(regions)
        print(len(regions))


if __name__ == "__main__":
    get_train_data()