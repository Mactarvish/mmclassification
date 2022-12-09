import torch
import torch.nn as nn

from ..builder import BACKBONES
from .base_backbone import BaseBackbone


@BACKBONES.register_module()
class HandCNNLSTM(BaseBackbone):
    def __init__(self, num_classes, num_frames, fea_num=21*3):
        super().__init__()
        self.num_frames = num_frames
        self.fea_num = fea_num
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, stride=1, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.dropout = nn.Dropout2d(0.3)

        self.lstm1 = nn.LSTM(input_size=64*fea_num, hidden_size=128, num_layers=1)
        self.lstm2 = nn.LSTM(input_size=128, hidden_size=64, num_layers=1)
        self.fc = nn.Linear(in_features=64, out_features=32)
        self.relu2 = nn.ReLU()
        self.head = nn.Linear(in_features=32, out_features=num_classes)
        self.flatten = nn.Flatten(-2, -1)

    def forward(self, x):
        x = x.reshape([x.shape[0], 1, self.num_frames, self.fea_num]) # 挤出一个C轴用于conv2d
        x = self.conv1(x)
        # x = self.bn1(x)
        x = self.relu1(x)
        x = self.pool(x)
        # x = self.dropout(x)

        x = self.flatten(x.permute(2, 0, 1, 3))
        x, (h, c) = self.lstm1(x) # D N C
        x, (h, c) = self.lstm2(x)
        x = x[-1] # 最后一个LSTM只要窗口中最后一个特征的输出
        x = self.fc(x)
        x = self.relu2(x)
        x = self.head(x)
        x = nn.Softmax(dim=1)(x)
        return x
