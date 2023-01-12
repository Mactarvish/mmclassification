import torch
import torch.nn as nn

from ..builder import BACKBONES
from .base_backbone import BaseBackbone


@BACKBONES.register_module()
class HandCNNLSTM(BaseBackbone):
    def __init__(self, num_frames, feat_num=21*3):
        super().__init__()
        self.num_frames = num_frames
        self.feat_num = feat_num
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, stride=1, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.dropout = nn.Dropout2d(0.3)

        self.lstm1 = nn.LSTM(input_size=64*feat_num, hidden_size=128, num_layers=1)
        self.lstm2 = nn.LSTM(input_size=128, hidden_size=64, num_layers=1)
        self.flatten = nn.Flatten(-2, -1)

    def forward(self, x):
        x = x.reshape([x.shape[0], 1, self.num_frames, self.feat_num]) # 挤出一个C轴用于conv2d
        x = self.conv1(x)
        # x = self.bn1(x)
        x = self.relu1(x)
        x = self.pool(x)
        # x = self.dropout(x)

        x = self.flatten(x.permute(2, 0, 1, 3))
        x, (h, c) = self.lstm1(x) # D N C
        x, (h, c) = self.lstm2(x)
        return x


@BACKBONES.register_module()
class HandCNNLSTM_V2(BaseBackbone):
    def __init__(self, num_frames, feat_num=21*3):
        super().__init__()
        self.num_frames = num_frames
        self.feat_num = feat_num

        self.mlp_block = nn.Sequential(
        nn.Linear(self.feat_num, 128),
        nn.ReLU(),
        nn.Linear(128, 256),
        nn.ReLU())

        self.lstm1 = nn.LSTM(input_size=256, hidden_size=128, num_layers=2)
        self.lstm2 = nn.LSTM(input_size=128, hidden_size=64, num_layers=2)

    def forward(self, x):
        # 注意必须是new_first
        x = x.reshape([x.shape[0], self.num_frames, self.feat_num]) # B D C
        
        x = self.mlp_block(x)
        x = x.permute(1, 0, 2)

        x, (h, c) = self.lstm1(x) # D N C
        x, (h, c) = self.lstm2(x)
        return x

    # attention
    # def forward(self,text, seq_len):
    #     emb = self.embedding(text)  # [batch_size, seq_len, embeding]=[128, 32, 300]
    #     H, _ = self.lstm(emb)  # [batch_size, seq_len, hidden_size * num_direction]=[128, 32, 256]

    #     M = self.tanh1(H)  # [128, 32, 256]
    #     # M = torch.tanh(torch.matmul(H, self.u))
    #     alpha = F.softmax(paddle.matmul(M, self.w), axis=1).unsqueeze(-1)  # [128, 32, 1]
    #     out = H * alpha  # [128, 32, 256]
    #     out = paddle.sum(out, 1)  # [128, 256]
    #     out = F.relu(out)
    #     out = self.fc1(out)
    #     out = self.fc(out)  # [128, 64]
    #     return out