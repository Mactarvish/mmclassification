# Copyright (c) OpenMMLab. All rights reserved.
from .cls_head import ClsHead
from .conformer_head import ConformerHead
from .deit_head import DeiTClsHead
from .efficientformer_head import EfficientFormerClsHead
from .linear_head import LinearClsHead
from .multi_label_csra_head import CSRAClsHead
from .multi_label_head import MultiLabelClsHead
from .multi_label_linear_head import MultiLabelLinearClsHead
from .stacked_head import StackedLinearClsHead
from .vision_transformer_head import VisionTransformerClsHead

from .dummy_head import DummyHead
from .lstm_head import LSTMHead

__all__ = [
    'ClsHead', 'LinearClsHead', 'StackedLinearClsHead', 'MultiLabelClsHead',
    'MultiLabelLinearClsHead', 'VisionTransformerClsHead', 'DeiTClsHead',
    'ConformerHead', 'EfficientFormerClsHead', 'CSRAClsHead',
    
    'DummyHead', 'LSTMHead',
]
