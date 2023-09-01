import os
import random

import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm
from copy import deepcopy
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.utils.data.sampler import RandomSampler
from torchvision.transforms import transforms
from torch.optim.lr_scheduler import ReduceLROnPlateau

class SegDataset(Dataset):
  def __init__(self, src_folder, mask_folder, base_name_list, transform, device):
    self.base_name_list = base_name_list
    self.src_folder = src_folder
    self.mask_folder = mask_folder
    self.transform = transform
    self.device = device

  def __len__(self):
    return len(self.base_name_list)

  def __getitem__(self, idx):
    src_path = os.path.join(self.src_folder, self.base_name_list[idx])
    src_image = Image.open(src_path)
    src_image = self.transform(src_image)
    mask_path = os.path.join(self.mask_folder, self.base_name_list[idx])
    mask_image = Image.open(mask_path)
    mask_image = self.transform(mask_image)
    return src_image.to(self.device), mask_image.to(self.device)


class Decoder(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super(Decoder, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv_relu = nn.Sequential(
            nn.Conv2d(middle_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x1 = torch.cat((x1, x2), dim=1)
        x1 = self.conv_relu(x1)
        return x1


class UNet(nn.Module):
    def contracting_block(self, in_channels, out_channels, kernel_size=3):
        block = torch.nn.Sequential(
            torch.nn.Conv2d(kernel_size=kernel_size, in_channels=in_channels, out_channels=out_channels),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.Conv2d(kernel_size=kernel_size, in_channels=out_channels, out_channels=out_channels),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(out_channels),
        )
        return block

    def expansive_block(self, in_channels, mid_channel, out_channels, kernel_size=3):
        block = torch.nn.Sequential(
            torch.nn.Conv2d(kernel_size=kernel_size, in_channels=in_channels, out_channels=mid_channel),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(mid_channel),
            torch.nn.Conv2d(kernel_size=kernel_size, in_channels=mid_channel, out_channels=mid_channel),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(mid_channel),
            torch.nn.ConvTranspose2d(in_channels=mid_channel, out_channels=out_channels, kernel_size=3, stride=2,
                                     padding=1, output_padding=1)
        )
        return block

    def final_block(self, in_channels, mid_channel, out_channels, kernel_size=3):
        block = torch.nn.Sequential(
            torch.nn.Conv2d(kernel_size=kernel_size, in_channels=in_channels, out_channels=mid_channel),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(mid_channel),
            torch.nn.Conv2d(kernel_size=kernel_size, in_channels=mid_channel, out_channels=mid_channel),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(mid_channel),
            torch.nn.Conv2d(kernel_size=kernel_size, in_channels=mid_channel, out_channels=out_channels, padding=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(out_channels),
        )
        return block

    def __init__(self, in_channel, out_channel):
        super(UNet, self).__init__()
        # Encode
        self.conv_encode1 = self.contracting_block(in_channels=in_channel, out_channels=64)
        self.conv_maxpool1 = torch.nn.MaxPool2d(kernel_size=2)
        self.conv_encode2 = self.contracting_block(64, 128)
        self.conv_maxpool2 = torch.nn.MaxPool2d(kernel_size=2)
        self.conv_encode3 = self.contracting_block(128, 256)
        self.conv_maxpool3 = torch.nn.MaxPool2d(kernel_size=2)
        # Bottleneck
        self.bottleneck = torch.nn.Sequential(
            torch.nn.Conv2d(kernel_size=3, in_channels=256, out_channels=512),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(512),
            torch.nn.Conv2d(kernel_size=3, in_channels=512, out_channels=512),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(512),
            torch.nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=3, stride=2, padding=1,
                                     output_padding=1)
        )
        # Decode
        self.conv_decode3 = self.expansive_block(512, 256, 128)
        self.conv_decode2 = self.expansive_block(256, 128, 64)
        self.final_layer = self.final_block(128, 64, out_channel)

    def crop_and_concat(self, upsampled, bypass, crop=False):
        if crop:
            c = (bypass.size()[2] - upsampled.size()[2]) // 2
            bypass = F.pad(bypass, (-c, -c, -c, -c))
        return torch.cat((upsampled, bypass), 1)

    def forward(self, x):
        # Encode
        encode_block1 = self.conv_encode1(x)
        encode_pool1 = self.conv_maxpool1(encode_block1)
        encode_block2 = self.conv_encode2(encode_pool1)
        encode_pool2 = self.conv_maxpool2(encode_block2)
        encode_block3 = self.conv_encode3(encode_pool2)
        encode_pool3 = self.conv_maxpool3(encode_block3)
        # Bottleneck
        bottleneck1 = self.bottleneck(encode_pool3)
        # Decode
        ##print(x.shape, encode_block1.shape, encode_block2.shape, encode_block3.shape, bottleneck1.shape)
        ##print('Decode Block 3')
        ##print(bottleneck1.shape, encode_block3.shape)
        decode_block3 = self.crop_and_concat(bottleneck1, encode_block3, crop=True)
        ##print(decode_block3.shape)
        ##print('Decode Block 2')
        cat_layer2 = self.conv_decode3(decode_block3)
        ##print(cat_layer2.shape, encode_block2.shape)
        decode_block2 = self.crop_and_concat(cat_layer2, encode_block2, crop=True)
        cat_layer1 = self.conv_decode2(decode_block2)
        ##print(cat_layer1.shape, encode_block1.shape)
        ##print('Final Layer')
        ##print(cat_layer1.shape, encode_block1.shape)
        decode_block1 = self.crop_and_concat(cat_layer1, encode_block1, crop=True)
        ##print(decode_block1.shape)
        final_layer = self.final_layer(decode_block1)
        ##print(final_layer.shape)
        return final_layer

# class UNet(nn.Module):
#     def __init__(self, n_class):
#         super().__init__()
#
#         self.base_model = torchvision.models.resnet18(False)
#         self.base_layers = list(self.base_model.children())
#         self.layer1 = nn.Sequential(
#             nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False),
#             self.base_layers[1],
#             self.base_layers[2])
#         self.layer2 = nn.Sequential(*self.base_layers[3:5])
#         self.layer3 = self.base_layers[5]
#         self.layer4 = self.base_layers[6]
#         self.layer5 = self.base_layers[7]
#         self.decode4 = Decoder(512, 256 + 256, 256)
#         self.decode3 = Decoder(256, 256 + 128, 256)
#         self.decode2 = Decoder(256, 128 + 64, 128)
#         self.decode1 = Decoder(128, 64 + 64, 64)
#         self.decode0 = nn.Sequential(
#             nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
#             nn.Conv2d(64, 32, kernel_size=3, padding=1, bias=False),
#             nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False)
#         )
#         self.conv_last = nn.Conv2d(64, n_class, 1)
#
#     def forward(self, input):
#         e1 = self.layer1(input)  # 64,128,128
#         e2 = self.layer2(e1)  # 64,64,64
#         e3 = self.layer3(e2)  # 128,32,32
#         e4 = self.layer4(e3)  # 256,16,16
#         f = self.layer5(e4)  # 512,8,8
#         d4 = self.decode4(f, e4)  # 256,16,16
#         d3 = self.decode3(d4, e3)  # 256,32,32
#         d2 = self.decode2(d3, e2)  # 128,64,64
#         d1 = self.decode1(d2, e1)  # 64,128,128
#         d0 = self.decode0(d1)  # 64,256,256
#         out = self.conv_last(d0)  # 1,256,256
#         return out


def flatten(tensor):
    """Flattens a given tensor such that the channel axis is first.
    The shapes are transformed as follows:
       (N, C, D, H, W) -> (C, N * D * H * W)
    """

    C = tensor.size(1)  # 获得图像的维数
    # new axis order
    axis_order = (1, 0) + tuple(range(2, tensor.dim()))
    # Transpose: (N, C, D, H, W) -> (C, N, D, H, W)
    transposed = tensor.permute(axis_order)  # 将维数的数据转换到第一位
    # Flatten: (C, N, D, H, W) -> (C, N * D * H * W)
    return transposed.contiguous().view(C, -1)


class DiceLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.epsilon = 1e-5

    def forward(self, output, target):
        target = torch.squeeze(target, dim=1)
        # print(output.size())
        # print(target.size())
        assert output.size() == target.size(), "'input' and 'target' must have the same shape"
        output = F.softmax(output, dim=1)
        output = flatten(output)
        target = flatten(target)
        # intersect = (output * target).sum(-1).sum() + self.epsilon
        # denominator = ((output + target).sum(-1)).sum() + self.epsilon

        intersect = (output * target).sum(-1)
        denominator = (output + target).sum(-1)
        dice = intersect / denominator
        dice = torch.mean(dice)
        return 1 - dice
        # return 1 - 2. * intersect / denominator

def train_model():
    folder = 'lung_segment/src_image'
    model_name = 'lung_seg_unet.pt'
    train_size = [284, 284]
    data_transform = transforms.Compose([transforms.Resize(train_size), transforms.ToTensor()])
    batch_size = 8

    png_files = [os.path.basename(file) for file in os.listdir(folder) if file.lower().endswith(".png")]
    random.shuffle(png_files)
    total_num = len(png_files)
    test_num = int(total_num*0.1)
    validate_num = int(total_num*0.9*0.2)
    validate_list = png_files[test_num:test_num+validate_num]
    train_list = png_files[test_num+validate_num:]

    # 使用自己的数据集
    # 如果GPU可用，利用GPU进行训练
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dataset = SegDataset('lung_segment/src_image', 'lung_segment/mask_image', train_list, transform=data_transform, device=device)
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=3,
                                  persistent_workers=True, prefetch_factor=3)

    validate_dataset = SegDataset('lung_segment/src_image', 'lung_segment/mask_image', validate_list, transform=data_transform, device=device)
    validate_dataloader = DataLoader(dataset=validate_dataset, batch_size=batch_size)
    validate_length = len(validate_dataloader)
    # net = ViTNet((3, 512, 512), n_patches=32, n_blocks=3, hidden_d=32, n_heads=16, out_d=1).to(device)
    # net = timm.create_model('vit_base_resnet50_384', pretrained=False, num_classes=1)
    net = UNet(1, 1)
    # net = Network(num_classes=1)
    if os.path.exists(model_name):
        net = torch.load(model_name)
    net = net.to(device)
    loss_fn = nn.CrossEntropyLoss()
    learning_rate = 0.01
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
            # targets = targets.unsqueeze(1)
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

def check_model():
    model = torch.load('lung_seg_unet.pt')  # 替换为实际的模型路径
    model.eval()
    folder = 'lung_segment/src_image'
    png_files = [os.path.basename(file) for file in os.listdir(folder) if file.lower().endswith(".png")]
    train_size = [512, 512]
    data_transform = transforms.Compose([transforms.Resize(train_size), transforms.ToTensor()])
    for i in range(len(png_files)):
        basename = png_files[i]
        src_path = os.path.join(folder, basename)
        print(src_path)
        src_image = Image.open(src_path)
        src_image = data_transform(src_image)
        # 使用模型进行预测
        with torch.no_grad():
            output = model(src_image.to('cuda').unsqueeze(0))

        segmentation_mask = output.argmax(dim=1).squeeze().cpu().to(torch.float32)*255
        to_pil = transforms.ToPILImage()
        image = to_pil(segmentation_mask)
        image.show()


if __name__ == '__main__':
    # get_append_train_data()
    train_model()
    # check_model()