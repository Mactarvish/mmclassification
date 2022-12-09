import torch.nn as nn
import torch.nn.functional as F

from ..builder import HEADS
from .cls_head import ClsHead


@HEADS.register_module()
class DummyHead(ClsHead):
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
                 init_cfg=dict(),
                 *args,
                 **kwargs):
        super(DummyHead, self).__init__(init_cfg=init_cfg, *args, **kwargs)

    def pre_logits(self, x):
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
        losses = self.loss(x, gt_label, **kwargs)
        return losses
