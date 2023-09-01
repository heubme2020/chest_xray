import os
import random
import shutil

import pandas as pd
import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
from torch.optim import Adam
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np
from torchvision.transforms import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from PIL import Image
import timm

import cv2

def get_test_data(train_folder):
    suffix = 'png'
    father_path = os.path.dirname(train_folder)
    child_folders = [f.path for f in os.scandir(train_folder) if f.is_dir()]
    test_folder = father_path + '/test'
    os.makedirs(test_folder, exist_ok=True)
    for i in range(len(child_folders)):
        child_folder = child_folders[i]
        child_files = [os.path.join(child_folder, file) for file in os.listdir(child_folder) if
                       file.lower().endswith(suffix)]
        random.shuffle(child_files)
        move_num = int(0.1*len(child_files))
        #创建对应test文件夹
        child_folder = child_folder.replace('train', 'test')
        os.makedirs(child_folder, exist_ok=True)
        #move部分图片到test文件夹
        for j in range(move_num):
            train_child_file = child_files[j]
            test_child_file = train_child_file.replace('train', 'test')
            shutil.move(train_child_file, test_child_file)
            
def get_validate_data(train_folder):
    suffix = 'png'
    father_path = os.path.dirname(train_folder)
    child_folders = [f.path for f in os.scandir(train_folder) if f.is_dir()]
    validate_folder = father_path + '/validate'
    os.makedirs(validate_folder, exist_ok=True)
    for i in range(len(child_folders)):
        child_folder = child_folders[i]
        child_files = [os.path.join(child_folder, file) for file in os.listdir(child_folder) if
                       file.lower().endswith(suffix)]
        random.shuffle(child_files)
        move_num = int(0.2*len(child_files))
        #创建对应validate文件夹
        child_folder = child_folder.replace('train', 'validate')
        os.makedirs(child_folder, exist_ok=True)
        #move部分图片到validate文件夹
        for j in range(move_num):
            train_child_file = child_files[j]
            validate_child_file = train_child_file.replace('train', 'validate')
            shutil.move(train_child_file, validate_child_file)
# 
class Network_256(nn.Module):
    def __init__(self, num_classes=10):
        super(Network_256, self).__init__()
        self.feature_extraction = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4, padding=2, bias=False),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0),
            nn.Conv2d(in_channels=96, out_channels=192, kernel_size=5, stride=1, padding=2, bias=False),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0),
            nn.Conv2d(in_channels=192, out_channels=384, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0),
        )

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features=256 * 7 * 7, out_features=256),
            nn.GELU(),
            nn.Linear(in_features=256, out_features=num_classes),
        )

    def forward(self, x):
        x = self.feature_extraction(x)
        x = x.view(x.size(0), 256 * 7 * 7)
        x = self.classifier(x)
        return x

class Network(nn.Module):
    def __init__(self, num_classes=10):
        super(Network, self).__init__()
        self.feature_extraction = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4, padding=2, bias=False),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0),
            nn.Conv2d(in_channels=96, out_channels=192, kernel_size=5, stride=1, padding=2, bias=False),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0),
            nn.Conv2d(in_channels=192, out_channels=384, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0),
        )

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features=256 * 15 * 15, out_features=1024),
            nn.GELU(),
            nn.Linear(in_features=1024, out_features=num_classes),
        )

    def forward(self, x):
        x = self.feature_extraction(x)
        x = x.view(x.size(0), 256 * 15 * 15)
        x = self.classifier(x)
        return x

def train_model():
    folder = 'lung_opacity'
    model_name = 'opacity_densenet121.pt'
    train_size = [512, 512]
    data_transform = transforms.Compose([transforms.Resize(train_size), transforms.ToTensor()])
    batch_size = 4
    # 使用自己的数据集
    train_dataset = datasets.ImageFolder(root=folder + '/train', transform=data_transform)
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=3,
                                  persistent_workers=True, prefetch_factor=3)

    validate_dataset = datasets.ImageFolder(root=folder + '/validate', transform=data_transform)
    validate_dataloader = DataLoader(dataset=validate_dataset, batch_size=batch_size)
    validate_length = len(validate_dataloader)

    classes = validate_dataset.classes
    # 如果GPU可用，利用GPU进行训练
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = timm.create_model('densenet121', pretrained=False, num_classes=3).to(device)
    # net = Network(num_classes=len(classes)).to(device)
    if os.path.exists(model_name):
        net = torch.load(model_name)
    net = net.to(device)
    loss_fn = nn.CrossEntropyLoss()
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
            # 将图像和标签移动到指定设备上
            features = features.to(device)
            targets = targets.to(device)

            # 梯度清零，也就是把loss关于weight的导数变成0.
            optimizer.zero_grad()
            # 获取网络输出
            output = net(features)
            # 获取损失
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
                # 将图像和标签移动到指定设备上
                features = features.to(device)
                targets = targets.to(device)

                optimizer.zero_grad()
                # 获取网络输出
                output = net(features)
                # 获取损失
                loss = loss_fn(output, targets)
                total_validate_loss += loss/float(validate_length)
        print("Epoch: %d, validate loss: %1.5f" % (i, total_validate_loss))
        if total_validate_loss < min_validate_loss:
            min_validate_loss = total_validate_loss
            torch.save(net, model_name)
        scheduler.step(total_validate_loss)  # 在每个epoch结束时调用学习率调度器
        print(optimizer.state_dict()['param_groups'][0]['lr'])

def metric_model(test_folder, model_name):
    custom_transform = transforms.Compose([transforms.Resize([512, 512]), transforms.ToTensor()])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = timm.create_model('densenet121', pretrained=False, num_classes=3).to(device)
    if os.path.exists(model_name):
        model = torch.load(model_name)

    test_dataset = datasets.ImageFolder( root=test_folder, transform=custom_transform)
    classes = test_dataset.classes
    test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=True)
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    for features, targets in test_loader:
        # predictions = model.classifier(extract_features.to(device))
        predictions = model.forward(features.to(device))
        # 预测得分最高的那一类对应的输出score
        pred = torch.argmax(predictions).item()
        pred_class = predictions[:, pred]
        predictions = torch.argmax(predictions, dim=1)
        predictions = predictions.cpu().detach().numpy()[0]
        targets = targets.cpu().detach().numpy()[0]

        if targets == 0 and predictions == 0:
            TN += 1
        if targets == 0 and predictions == 1:
            FP += 1
        if targets == 1 and predictions == 1:
            TP += 1
        if targets == 1 and predictions == 0:
            FN += 1
    predict_num = TP + FP + TN + FN
    accuracy = float(TP + TN)/(TP + FP + TN + FN)
    precision = float(TP)/(TP + FP)
    recall = float(TP)/(TP + FN)
    F1 = 2*precision*recall/(precision + recall)
    print(predict_num)
    print('accuracy:' + str(accuracy))
    print('precision:' + str(precision))
    print('recall:' + str(recall))
    print('F1:' + str(F1))

def transform_image(image):
    transform = transforms.Compose([transforms.Resize((512, 512)), transforms.ToTensor()])
    return transform(image)

def get_append_train_data():
    model_name = 'view.pt'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Network(num_classes=2).to(device)
    if os.path.exists(model_name):
        model = torch.load(model_name)

    os.makedirs('view/append_train', exist_ok=True)
    os.makedirs('view/append_train/AP', exist_ok=True)
    os.makedirs('view/append_train/PA', exist_ok=True)
    image_folder = 'chest-xray-14/images'
    label_file = 'chest-xray-14/Data_Entry_2017.csv'
    label_data = pd.read_csv(label_file)
    label_data = label_data.sample(frac=1, random_state=None)
    label_data = label_data.reset_index(drop=True)
    sample_num = 0
    error_num = 0
    for index, row in label_data.iterrows():
        image_name = os.path.join(image_folder, row['Image Index'])
        if os.path.exists(image_name):
            image = Image.open(image_name)
            image = image.convert("RGB")
            image = transform_image(image)
            image = image.unsqueeze(0)
            predictions = model.forward(image.to(device))
            pred = torch.argmax(predictions).item()
            sample_num += 1
            if pred == 0:
                if row['View Position'] == 'PA':
                    error_num += 1
                    image_dst_name = 'view/append_train/PA/' + row['Image Index']
                    shutil.copy(image_name, image_dst_name)
            elif pred == 1:
                if row['View Position'] == 'AP':
                    error_num += 1
                    image_dst_name = 'view/append_train/AP/' + row['Image Index']
                    shutil.copy(image_name, image_dst_name)

        print(float(error_num)/sample_num)

def draw_CAM(model, img_path, save_path, transform=None, visual_heatmap=False):
    '''
    绘制 Class Activation Map
    :param model: 加载好权重的Pytorch model
    :param img_path: 测试图片路径
    :param save_path: CAM结果保存路径
    :param transform: 输入图像预处理方法
    :param visual_heatmap: 是否可视化原始heatmap（调用matplotlib）
    :return:
    '''
    # 图像加载&预处理
    img = Image.open(img_path).convert('RGB')
    if transform:
        img = transform(img)
    img = img.unsqueeze(0)

    # 获取模型输出的feature/score
    model.eval()
    features = model.features(img)
    output = model.classifier(features)

    # 为了能读取到中间梯度定义的辅助函数
    def extract(g):
        global features_grad
        features_grad = g

    # 预测得分最高的那一类对应的输出score
    pred = torch.argmax(output).item()
    pred_class = output[:, pred]

    features.register_hook(extract)
    pred_class.backward()  # 计算梯度

    grads = features_grad  # 获取梯度

    pooled_grads = torch.nn.functional.adaptive_avg_pool2d(grads, (1, 1))

    # 此处batch size默认为1，所以去掉了第0维（batch size维）
    pooled_grads = pooled_grads[0]
    features = features[0]
    # 512是最后一层feature的通道数
    for i in range(256):
        features[i, ...] *= pooled_grads[i, ...]

    # 以下部分同Keras版实现
    heatmap = features.detach().numpy()
    heatmap = np.mean(heatmap, axis=0)

    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)

    # 可视化原始热力图
    if visual_heatmap:
        plt.matshow(heatmap)
        plt.show()

    img = cv2.imread(img_path)  # 用cv2加载原始图像
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))  # 将热力图的大小调整为与原始图像相同
    heatmap = np.uint8(255 * heatmap)  # 将热力图转换为RGB格式
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)  # 将热力图应用于原始图像
    superimposed_img = heatmap * 0.4 + img  # 这里的0.4是热力图强度因子
    cv2.imwrite(save_path, superimposed_img)  # 将图像保存到硬盘


def split_train_data(folder):
    get_test_data(folder)
    get_validate_data(folder)
if __name__ == '__main__':
    # model = Network()
    # print(model.eval())
    # Define the loss function with Classification Cross-Entropy loss and an optimizer with Adam optimizer
    # loss_fn = nn.CrossEntropyLoss()
    # optimizer = Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
    # split_train_data('lung_opacity/train')
    # train_model()
    metric_model('lung_opacity/test', 'opacity_densenet121.pt')
    # get_append_train_data()

