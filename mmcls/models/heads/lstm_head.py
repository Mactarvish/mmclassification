import torch.nn as nn
import torch.nn.functional as F

from ..builder import HEADS
from .cls_head import ClsHead

import torch
import numpy as np
from ..utils import is_tracing

@HEADS.register_module()
class LSTMHead(ClsHead):
    """Linear classifier head.

    Args:
        num_classes (int): Number of categories excluding the background
            category.
        in_channels (int): Number of channels in the input feature map.
        init_cfg (dict | optional): The extra init config of layers.
            Defaults to use dict(type='Normal', layer='Linear', std=0.01).
    """

    def __init__(self,
                #  init_cfg=dict(type='Normal', layer='Linear', std=0.01),
                 num_classes,
                 only_last,
                 init_cfg=dict(),
                 *args,
                 **kwargs):
        super(LSTMHead, self).__init__(init_cfg=init_cfg, *args, **kwargs)
        self.num_classes = num_classes
        self.only_last = only_last
        self.fc = nn.Linear(in_features=64, out_features=32)
        self.relu2 = nn.ReLU()
        self.head = nn.Linear(in_features=32, out_features=num_classes)


    def pre_logits(self, x):
        if self.only_last:
            i = x.size(0) - 1
            x = x[i] # 最后一个LSTM只要窗口中最后一个特征的输出
        x = self.fc(x)
        x = self.relu2(x)
        x = self.head(x)
        x = nn.Softmax(dim=-1)(x)
        if not self.only_last:
            x = x.transpose(1, 0)
        return x

    def simple_test(self, x, softmax=True, post_process=True):
        """Inference without augmentation.

        Args:
            x (tuple[Tensor]): The input features.
                Multi-stage inputs are acceptable but only the last stage will
                be used to classify. The shape of every item should be
                ``(num_samples, in_channels)``.
            softmax (bool): Whether to softmax the classification score.
            post_process (bool): Whether to do post processing the
                inference results. It will convert the output to a list.

        Returns:
            Tensor | list: The inference results.

                - If no post processing, the output is a tensor with shape
                  ``(num_samples, num_classes)``.
                - If post processing, the output is a multi-dimentional list of
                  float and the dimensions are ``(num_samples, num_classes)``.
        """
        cls_score = self.pre_logits(x)

        pred = cls_score

        if post_process:
            return self.post_process(pred)
        else:
            return pred

    def forward_train(self, x, gt_label, **kwargs):
        x = self.pre_logits(x)
        losses = self.loss(x, gt_label, **kwargs)
        return losses
    
    def post_process(self, pred):
        on_trace = is_tracing()
        if torch.onnx.is_in_onnx_export() or on_trace:
            return pred
        pred = list(pred.detach().cpu().numpy())
        if not self.only_last:
            # 除了0以外，最多且超过2个的元素
            pred_labels = []
            for e in pred:
                class_count = e.argmax(axis=1)
                maxc = -1
                maxi = 0
                for i in range(pred[0].shape[-1]):
                    if (class_count == i).sum() > maxc:
                        maxc = (class_count == i).sum()
                        maxi = i
                if maxc > 2:
                    pred_labels.append(maxi)
                else:
                    pred_labels.append(0)
            return np.eye(self.num_classes)[pred_labels]

        return pred

