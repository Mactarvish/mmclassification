import torch
import random
import numpy as np
from collections import defaultdict

from tqdm import tqdm
import datetime
import sys
import os
import glob
import json
import cv2

from tabulate import tabulate

from .base_dataset import BaseDataset
from .builder import DATASETS
from abc import ABCMeta, abstractmethod


from utils.visualize_hand_pose import vis_hand_pose_3d, create_gif

LABELS = ["none", "up", "down", "left", "right"][:3]
if len(LABELS) == 3:
    print("不考虑左右")
    print("不考虑左右")
    print("不考虑左右")
    print("不考虑左右")
    print("不考虑左右")
    print("不考虑左右") 
    
    
class Frame():
    def __init__(self, labelme_path, landmark, label, num_frame) -> None:
        # 存储单帧的全部信息
        assert labelme_path.endswith('.json'), labelme_path
        assert os.path.exists(labelme_path), labelme_path
        assert label in LABELS, label
        assert isinstance(landmark, np.ndarray), landmark

        self.labelme_path = labelme_path
        self.depth_path = labelme_path.replace(".json", ".png").replace("merge_result", "depth")
        self.landmark = landmark
        self.embedding = self.landmark
        self.num_frame = num_frame
        self.label = label


@DATASETS.register_module()
class HandSlideDataset(BaseDataset, metaclass=ABCMeta):
    def __init__(self,
                 src_dir,
                 pipeline,
                 duration,
                 num_keypoints,
                 *,
                 single_finger,
                 test_mode,
                 visualize):
        # assert "backup" not in src_dir, src_dir
        # assert "slide/" in src_dir, src_dir
        self.src_dir = src_dir
        
        self.duration = duration
        self.single_finger = single_finger
        self.frames = self.parse_jsons_to_frames(self.src_dir, num_keypoints)
        # 对帧序列进行清洗，删除重复动作帧等无效帧
        self.frames = self.preprocess_frames()

        cache_name = '_'.join(self.src_dir.split("slide/")[-1].split('/')) + ".pkl"
        cache_path = os.path.join("pgs", cache_name)
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        # debug模式不加载缓存
        if os.path.exists(cache_path) and not sys.gettrace():
            cache = torch.load(cache_path)
            self.samples = cache["samples"]
            print(f"加载samples缓存 {cache_path} ：生成时间 {cache['modify_time']}")
        else:
            # 无缓存文件，重新生成样本
            self.samples = self.generate_samples()
            print(f"生成缓存到 {cache_path}")
            torch.save({"samples": self.samples,
                        "modify_time": datetime.datetime.now(),
                        "duration": self.duration},
                       cache_path)
        
        # 统计样本分布
        self.sample_statics = self.static_sample_dist()
        
        # 训练阶段进行优质样本挖掘
        if not test_mode:
            self.samples = self.mine_good_cases()
        
        # 样本可视化
        if visualize:
            self.visualize_samples()
        super().__init__(self.src_dir, pipeline)
                    
    def parse_jsons_to_frames(self, src_dir, num_keypoints):
        src_json_paths = sorted(glob.glob(os.path.join(self.src_dir, "**", "merge_result", "*.json"), recursive=True))
        # 新的在最前头
        src_json_paths = src_json_paths[::-1]
        assert len(src_json_paths) != 0, self.src_dir
        # 首先把没有框的样本剔除，保持后续遍历时索引的连续性
        src_json_paths = list(filter(lambda p: "kp" in json.load(open(p, 'r')), src_json_paths))
        
        # 逐个json解析成Frame
        frames = []
        for i, src_json_path in enumerate(src_json_paths):
            json_dict = json.load(open(src_json_path, 'r'))
            # assert "bbox" in json_dict
            assert "kp" in json_dict
            assert "on_table" in json_dict

            # 解析landmark
            if self.single_finger:
                landmark = np.zeros((4 , 3), dtype=np.float32)
                landmark[0] = json_dict["kp"][1][:3]
                landmark[1] = json_dict["kp"][6][:3]
                landmark[2] = json_dict["kp"][11][:3]
                landmark[3] = json_dict["kp"][15][:3]
            else:
                landmark = np.zeros((num_keypoints , 3), dtype=np.float32)
                for j in range(num_keypoints):
                    landmark[j] = json_dict["kp"][j][:3]
                
            # 确定当前帧的标签
            if json_dict["on_table"]:
                for e in LABELS:
                    if '/' + e in src_json_path:
                        label = e
            else:
                label = "none"
                
            frames.append(Frame(src_json_path, landmark, label, i))
            
        return frames

    def load_annotations(self):
        data_infos = []
        data_infos = self.samples
        return data_infos
        
    # 重写获取标签的函数
    def get_gt_labels(self):
        """Get all ground-truth labels (categories).

        Returns:
            np.ndarray: categories for all images.
        """
        gt_labels = np.array([LABELS.index(data['patch_label']) for data in self.data_infos])
        return gt_labels

    def generate_samples(self):
        '''
        根据帧序列生成样本
        '''
        samples = []
        for i in tqdm(range(len(self.frames) - self.duration + 1)):
            sample = self.generate_single_sample([i, i + self.duration - 1])
            samples.append(sample)
        return samples

    def generate_soft_label(self, prob, label):
        assert prob <= 1, prob
        sl = np.zeros(len(LABELS), dtype=np.float)
        sl[LABELS.index(label)] = prob
        sl[0] = 1 - prob
        return sl
    
    def preprocess_frames(self):
        # 1. 删除连续的微小差异有效动作帧
        IGNORED_FRAME_DIFF_THRESHOLD = 10
        last_valid_frame_index = 0
        to_be_removed_indexes = []
        for i in range(1, len(self.frames)):
            # 如果当前帧和上一有效帧都是有效动作帧，那么计算帧差并根据帧差决定当前帧的裁决
            if self.frames[i].label != "none" :
                if self.frames[last_valid_frame_index].label != "none":
                    dist = np.linalg.norm(self.frames[i].embedding - self.frames[last_valid_frame_index].embedding)
                    # 帧差太小，删除当前帧
                    if dist < IGNORED_FRAME_DIFF_THRESHOLD:
                        to_be_removed_indexes.append(i)
                        if sys.gettrace():
                            print(dist)
                            print(self.frames[i].depth_path.replace("depth", "normal"))
                            print(self.frames[i - 1].depth_path.replace("depth", "normal"))
                            print(self.frames[i].landmark)
                            print(self.frames[i - 1].landmark)
                            import shutil
                            shutil.copy(self.frames[i].depth_path.replace("depth", "normal"), "pgs/f1.png")
                            shutil.copy(self.frames[i - 1].depth_path.replace("depth", "normal"), "pgs/f2.png")
                            print()
            # 当前帧未被移除，说明当前帧有效，下次算帧差跟当前帧算
            if len(to_be_removed_indexes) == 0 or to_be_removed_indexes[-1] != i:
                last_valid_frame_index = i

        kept_frames = []
        for i in range(len(self.frames)):
            if i not in to_be_removed_indexes:
                kept_frames.append(self.frames[i])
        self.frames = kept_frames
        # 2. 删除孤立动作帧
        to_be_removed_indexes = []
        for i in range(len(self.frames)):
            last_none = i == 0 or self.frames[i - 1].label == "none"
            next_none = i == len(self.frames) - 1 or self.frames[i + 1].label == "none"
            if self.frames[i].label != "none" and last_none and next_none:
                to_be_removed_indexes.append(i)
                
        kept_frames = []
        for i in range(len(self.frames)):
            if i not in to_be_removed_indexes:
                kept_frames.append(self.frames[i])
        self.frames = kept_frames

            
        
                
        return self.frames

    @abstractmethod
    def generate_single_sample(self, indexes):
        return dict()

    def __str__(self):
        dataset_name = '_'.join(self.src_dir.split("slide/")[-1].split('/'))
        s = '\n'.join([dataset_name, self.sample_statics])
        return s

    @abstractmethod
    def static_sample_dist(self):
        print("原始数据统计：")
        begin = 0
        cur_label = "none"
        label_durations = defaultdict(list)
        for i in range(len(self.frames)):
            if i > 0 and self.frames[i - 1].label != self.frames[i].label:
                # 新动作开始
                if self.frames[i].label != "none":
                    begin = i
                    cur_label = self.frames[i].label
                # 动作结束
                else:
                    duration = i - begin
                    label_durations[cur_label].append(duration)

        for k in label_durations:
            label_durations[k] = sorted(label_durations[k])
        
        raw_table = []
        # for k in ["none", "up", "down"]:
        for k in ["up", "down"]:
            row = [k, len(label_durations[k]), sum(label_durations[k]) / len(label_durations[k]), min(label_durations[k]), max(label_durations[k])]
            raw_table.append(row)
        table = tabulate(
        raw_table,
        headers=["类别", "数量", "平均时长", "最短时长", "最大时长"],
        tablefmt="pipe",
        numalign="left",
        stralign="center",
        )
        print(table)
            
        
        print("数据集训练样本统计：")
        label_durations.clear()
        for s in self.samples:
            m = 0
            while m < len(s["per_frame_label"]) and s["per_frame_label"][m] == s["per_frame_label"][0]:
                m += 1
            label_durations[s["patch_label"]].append(m)
        sample_table = []
        for k in ["none", "up", "down"]:
            row = [k, len(label_durations[k]), sum(label_durations[k]) / len(label_durations[k]), min(label_durations[k]), max(label_durations[k])]
            sample_table.append(row)
        table = tabulate(
        sample_table,
        headers=["类别", "数量", "平均时长", "最短时长", "最大时长"],
        tablefmt="pipe",
        numalign="left",
        stralign="center",
        )
        print(table)
        
        return table

    @abstractmethod
    def mine_good_cases(self):
        return self.samples
        
    def visualize_samples(self, max_vis=10):
        vis_samples = self.samples
        if max_vis != -1:
            vis_samples = random.sample(self.samples, max_vis)
        for s in tqdm(vis_samples):
            vis_nps = vis_hand_pose_3d(s["img"])
            if len(vis_nps) != 0:
                vis_nps = [vis_nps[0]] + vis_nps
                vis_name = os.path.splitext(os.path.basename(s["src_depth_paths"][0]))[0]
                vis_dir = "pgs/vis"
                os.makedirs(vis_dir, exist_ok=True)
                create_gif(vis_nps, os.path.join(vis_dir, vis_name + ".gif"))