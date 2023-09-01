import pandas as pd
import os
from torch.utils.data import Dataset, DataLoader
from densenet import DenseNet
import torch
from PIL import Image
from torchvision.transforms import transforms
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from classification import Network
import timm
import glob
import random
import pydicom
import uuid
from vitnet import ViTNet
from tqdm import tqdm

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
def read_txt_data(folder):
    txt_files = [os.path.abspath(os.path.join(folder, file)) for file in os.listdir(folder) if
                 file.lower().endswith('.txt')]
    data = {'image': [],
            'age': []}
    age_data = pd.DataFrame(data)
    for i in range(len(txt_files)):
        txt_file = txt_files[i]
        base_name = os.path.basename(txt_file)
        image_name = base_name.replace('txt', 'png')
        txt_data = read_txt(txt_file)
        age = txt_data[5].replace('Y', '')
        age = int(age)
        new_row = pd.DataFrame({'image': [image_name], 'age': [age]})
        age_data = pd.concat([age_data, new_row], ignore_index=True)
    print(age_data)
    age_data.to_csv('age_data_append.csv', index=False)

def patchify(images, n_patches):
    n, c, h, w = images.shape

    assert h == w, "Patchify method is implemented for square images only"

    patches = torch.zeros(n, n_patches ** 2, h * w * c // n_patches ** 2)
    patch_size = h // n_patches

    for idx, image in enumerate(images):
        for i in range(n_patches):
            for j in range(n_patches):
                patch = image[:, i * patch_size: (i + 1) * patch_size, j * patch_size: (j + 1) * patch_size]
                patches[idx, i * n_patches + j] = patch.flatten()
    return patches

class RegressionDataset(Dataset):
    def __init__(self, data, folder, transform=None, device=None):
        image_list = data['image'].tolist()
        age_list = data['age'].tolist()
        self.image_list = image_list
        self.age_list = age_list
        self.folder = folder
        self.transform = transform
        self.device = device

    def __len__(self):
        return len(self.age_list)

    def __getitem__(self, index):
        image_path = os.path.join(self.folder, self.image_list[index])
        age = float(self.age_list[index])
        age = torch.tensor(age)
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image.to(self.device), age.to(self.device)


def get_append_age_data():
    label_data = pd.read_csv('label.csv')
    label_data.drop('PatientID', axis=1, inplace=True)
    label_data.drop('ViewPosition', axis=1, inplace=True)
    label_data.drop('PatientSex', axis=1, inplace=True)
    label_data.rename(columns={'PngBasename': 'image'}, inplace=True)
    label_data.rename(columns={'PatientAge': 'age'}, inplace=True)
    label_data.to_csv('age/age_append_data.csv', index=False)

def merge_label():
    label_data = pd.read_csv('age/age_data.csv')
    label_append_data = pd.read_csv('age/age_append_data.csv')
    age_data = pd.concat([label_data, label_append_data], ignore_index=True)
    age_data = age_data.sample(frac=1, random_state=None)
    age_data = age_data.reset_index(drop=True)
    age_data.to_csv('age/age_data.csv', index=False)

def train_model():
    folder = 'age'
    model_name = 'age_resnet50.pt'
    train_size = [512, 512]
    data_transform = transforms.Compose([transforms.Resize(train_size), transforms.ToTensor()])
    batch_size = 8

    age_data = pd.read_csv(os.path.join(folder, 'age_data.csv'))
    age_data = age_data.sample(frac=1, random_state=1024)
    age_data = age_data.reset_index(drop=True)
    total_num = len(age_data)
    test_num = int(total_num*0.1)
    validate_num = int(total_num*0.9*0.2)
    age_data_test = age_data[:test_num]
    age_data_validate = age_data[test_num:test_num+validate_num]
    age_data_validate = age_data_validate.reset_index(drop=True)
    age_data_train = age_data[test_num+validate_num:]
    age_data_train = age_data_train.reset_index(drop=True)
    # 使用自己的数据集
    # 如果GPU可用，利用GPU进行训练
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dataset = RegressionDataset(age_data_train, 'age/train', transform=data_transform, device=device)
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=3,
                                  persistent_workers=True, prefetch_factor=3)

    validate_dataset = RegressionDataset(age_data_validate, 'age/train', transform=data_transform, device=device)
    validate_dataloader = DataLoader(dataset=validate_dataset, batch_size=batch_size)
    validate_length = len(validate_dataloader)
    # net = ViTNet((3, 512, 512), n_patches=32, n_blocks=3, hidden_d=32, n_heads=16, out_d=1).to(device)
    # net = timm.create_model('vit_base_resnet50_384', pretrained=False, num_classes=1)
    net = timm.create_model('resnet50d', pretrained=False, num_classes=1, global_pool='catavgmax')
    # net = Network(num_classes=1)
    if os.path.exists(model_name):
        net = torch.load(model_name)
    net = net.to(device)
    loss_fn = nn.L1Loss()
    learning_rate = 0.001
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=5, verbose=True)
    min_validate_loss = float("inf")
    # 训练轮数
    epoch = 100
    for i in range(epoch):
        mean_loss = 0
        step_num = 0
        net.train()  # 将网络设置为训练模式，当网络包含 Dropout, BatchNorm时必须设置，其他时候无所谓
        for (features, targets) in train_dataloader:
            # # 将图像和标签移动到指定设备上
            # features = features.to(device)
            # targets = targets.to(device)

            # 梯度清零，也就是把loss关于weight的导数变成0.
            optimizer.zero_grad()
            # 获取网络输出
            output = net(features)
            # 获取损失
            targets = targets.unsqueeze(1)
            loss = loss_fn(output, targets)
            mean_loss = (mean_loss*step_num + loss)/float(step_num + 1)
            step_num = step_num + 1
            loss.backward()  # calculates the loss of the loss function
            optimizer.step()  # improve from loss, i.e backprop
            print("Epoch: %d, train loss: %1.5f, mean loss: %1.5f, min loss: %1.5f" % (i, loss.item(), mean_loss,
                                                                                       min_validate_loss))
        # 将网络设置为测试模式，当网络包含 Dropout, BatchNorm时必须设置，其他时候无所谓
        net.eval()
        # 验证集部分
        total_validate_loss = 0
        with torch.no_grad():
            for (features, targets) in validate_dataloader:
                # # 将图像和标签移动到指定设备上
                # features = features.to(device)
                # targets = targets.to(device)

                optimizer.zero_grad()
                # 获取网络输出
                output = net(features)
                # 获取损失
                targets = targets.unsqueeze(1)
                loss = loss_fn(output, targets)
                total_validate_loss += loss/float(validate_length)
        print("Epoch: %d, validate loss: %1.5f" % (i, total_validate_loss))
        if total_validate_loss < min_validate_loss:
            min_validate_loss = total_validate_loss
            torch.save(net, model_name)
        scheduler.step(total_validate_loss)  # 在每个epoch结束时调用学习率调度器
        print(optimizer.state_dict()['param_groups'][0]['lr'])

def generate_uuid_string():
    return str(uuid.uuid4())

def get_append_train_data():
    model_name = 'age_resnet50.pt'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = timm.create_model('resnet50d', pretrained=False, num_classes=1, global_pool='catavgmax')
    if os.path.exists(model_name):
        model = torch.load(model_name)

    train_size = [512, 512]
    data_transform = transforms.Compose([transforms.Resize(train_size), transforms.ToTensor()])

    os.makedirs('age/append_train', exist_ok=True)
    image_folder = 'D:/Chexpert'
    label_file = os.path.join(image_folder, 'CheXpert-v1.0/train.csv')
    data = pd.read_csv(label_file)
    data = data[32768:]
    data = data.reset_index(drop=True)
    data = data.sample(frac=1, random_state=1024)
    data = data.reset_index(drop=True)
    data = data[:10000]
    label_data = pd.DataFrame(columns=['image', 'age'])
    total_num = 0
    error_num = 0
    for index, row in data.iterrows():
        frontal = row['Frontal/Lateral']
        if frontal == 'Lateral':
            continue
        image_name = os.path.join(image_folder, row['Path'])
        image_src = Image.open(image_name)
        image = image_src.convert("RGB")
        image = data_transform(image)
        image = image.unsqueeze(0)
        predictions = model.forward(image.to(device))
        predict = predictions.cpu().item()
        age = float(row['Age'])
        total_num += 1
        if abs(age-predict) > 7:
            print('predict:' + str(predict))
            print('age:' + str(age))
            print('diff:' + str(abs(age-predict)))
            uuid_str = generate_uuid_string()
            png_name = uuid_str + '.png'
            row_data = [png_name, age]
            label_data.loc[len(label_data)] = row_data
            png_name = os.path.join('age/append_train', png_name)
            image_src.save(png_name, optimize=True, compress_level=5, quality=85)
            error_num += 1
            print('error_rate:' + str(float(error_num)/float(total_num)))
    label_data.to_csv('age/age_append_data.csv', index=False)


if __name__ == '__main__':
    # get_append_train_data()
    merge_label()
    # train_model()