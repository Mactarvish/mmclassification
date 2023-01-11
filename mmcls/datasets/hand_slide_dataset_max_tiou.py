import torch
import os
import random
import numpy as np
from collections import defaultdict

from tqdm import tqdm
import datetime
import sys
import os
import glob
import json
import numpy as np
from tabulate import tabulate

from .hand_slide_dataset import HandSlideDataset, LABELS
from .builder import DATASETS


@DATASETS.register_module()
class HandSlideDatasetMaxTIOU(HandSlideDataset):
    def __init__(self,
                 src_dir,
                 pipeline,
                 duration,
                 num_keypoints,
                 single_finger,
                 test_mode,
                 visualize=False):
        super().__init__(src_dir, pipeline, duration, num_keypoints, single_finger=single_finger, test_mode=test_mode, visualize=visualize)

    # 重写获取标签的函数
    def get_gt_labels(self):
        """Get all ground-truth labels (categories).

        Returns:
            np.ndarray: categories for all images.
        """
        gt_labels = np.array([data['gt_label'] for data in self.data_infos]).argmax(axis=1)
        return gt_labels

    def generate_single_sample(self, indexes):
        # 1. 计算所有跟当前片段有重叠的有效动作片段(Valid Action Patches, VAP)
        begin = indexes[0]
        end = indexes[0]
        valid_action_patches = []
        # 确定遍历的起点：包含当前片段起点的有效动作片段的起点
        if self.frames[indexes[0]].label != "none":
            i = indexes[0]
            while i >= 0 and self.frames[i].label != "none":
                i -= 1
            begin = max(i, 0)

        last_valid = False
        for i in range(begin, len(self.frames)):
            # 当前帧有效
            if self.frames[i].label != "none":
                # 上一帧无效，说明是一个新的有效动作片段
                if not last_valid:
                    begin = i
                last_valid = True
            # 当前帧无效
            else:
                if last_valid:
                    end = i - 1
                    valid_action_patches.append([begin, end])
                last_valid = False
                # 出界，结束遍历
                if i > indexes[-1]:
                    break
        if i == len(self.frames):
            if last_valid:
                end = i - 1
                valid_action_patches.append([begin, end])
        best_score = 0
        best_label = "none"
        for k, vap in enumerate(valid_action_patches):
            # 计算vap和当前片段的tiou
            vap_label = self.frames[vap[0]].label
            inter = max(0, min(vap[-1], indexes[-1]) - max(vap[0], indexes[0]) + 1)
            union = vap[-1] - vap[0] + 1 + indexes[-1] - indexes[0] + 1 - inter
            tiou = inter / union
            score = tiou
            if score > best_score :
                # 当前片段的分数
                best_score = score
                best_label = vap_label
        soft_label = np.zeros(len(LABELS))
        # 具有最大tiou的vap的类别作为软标签的类别，分数作为软标签的分数
        soft_label[LABELS.index(best_label)] = best_score
        soft_label[0] = 1 - best_score

        frames = [self.frames[k] for k in range(indexes[0], indexes[1] + 1)]
        result = dict()
        result["src_depth_paths"]  =[f.depth_path for f in frames]
        result["img"] = np.array([f.embedding for f in frames])
        result["gt_label"] = soft_label
        result["per_frame_label"] = [LABELS.index(f.label) for f in frames]
        result["patch_label"] = LABELS[soft_label.argmax()]

        return result
    
    def mine_good_cases(self):
        return super().mine_good_cases()
    
    def static_sample_dist(self):
        return super().static_sample_dist()
    