import torch
import time
import os
import random
import numpy as np
from collections import defaultdict

from torch.utils.data.dataset import Dataset
from tqdm import tqdm
import datetime

from .utils.poseembedding import FullBodyPoseEmbedder

import os
import glob
import cv2
import json
import numpy as np

LABELS = ["none", "up", "down", "left", "right"][:3]
if len(LABELS) == 3:
    print("不考虑左右")
    print("不考虑左右")
    print("不考虑左右")
    print("不考虑左右")
    print("不考虑左右")
    print("不考虑左右")

args = None


class Frame():
    def __init__(self, labelme_path, landmark, label, num_frame) -> None:
        assert labelme_path.endswith('.json'), labelme_path
        assert os.path.exists(labelme_path), labelme_path
        assert label in LABELS, label
        assert isinstance(landmark, np.ndarray), landmark

        self.labelme_path = labelme_path
        self.depth_path = labelme_path.replace(".json", ".png").replace("infer_result", "depth")
        # assert os.path.exists(self.depth_path), self.depth_path
        self.landmark = landmark
        self.num_frame = num_frame
        self.label = label



def parse_on_table(src_dir):
    # 每个目录里只有一种动作
    if sum('/' + e in src_dir for e in LABELS) != 1:
        print("注意，无效目录：", src_dir)
        return [],[],[]
    
    for e in LABELS:
        if '/' + e in src_dir:
            action_name = e
    
    # action_json_dir = os.path.join(src_dir, "infer_result")
    action_json_dir = os.path.join(src_dir, "merge_result")
    assert os.path.exists(action_json_dir), action_json_dir

    src_json_paths = sorted(glob.glob(os.path.join(action_json_dir, "*.json")))
    # 首先把没有框的样本剔除，保持后续遍历时索引的连续性
    src_json_paths = list(filter(lambda p: "bbox" in json.load(open(p, 'r')), src_json_paths))
    
    frame_labels = []
    action_patches = []
    frame_paths = []
    begin = -1
    end = -1
    i = 0
    for i, src_json_path in enumerate(src_json_paths):
        frame_paths.append(src_json_path)
        json_dict = json.load(open(src_json_path, 'r'))
        assert "bbox" in json_dict
        assert "on_table" in json_dict
        on_table = json_dict["on_table"]
        frame_labels.append(action_name if on_table else "none")
        if (i == 0 or frame_labels[i - 1] == "none") and frame_labels[i] != "none":
            begin = i
        if frame_labels[i] == "none" and (i != 0 and frame_labels[i - 1] != "none"):
            end = i - 1
            # 从0开始算
            action_patches.append([begin, end, action_name])
    # 尾判
    if frame_labels[-1] != "none":
        end = i - 1
        action_patches.append([begin, end, action_name])
    
    return frame_paths, frame_labels, action_patches
    

def parse_raw_files_to_frames(src_dir, num_keypoint):
    # 从帧率解析文件中读取每一帧对应的动作类型，以及所有的标注动作片段
    frame_paths, frame_labels, action_patches = parse_on_table(src_dir)
    
    # 将每一帧解析成Frame
    frames = []
    for i in range(len(frame_paths)):
        src_json_path = frame_paths[i]
        json_dict = json.load(open(src_json_path, 'r'))
        landmark = np.zeros((21 , 3), dtype=np.float32)
        for j in range(num_keypoint):
            landmark[j] = json_dict["kp"][j][:3]
        frame = Frame(src_json_path, landmark, frame_labels[i], i)
        frames.append(frame)
    
    return frames


from .base_dataset import BaseDataset
from .builder import DATASETS

class Action():
    def __init__(self, begin, end,  frames, soft_label) -> None:
        self.begin = begin
        self.end = end
        self.frames = frames
        self.soft_label = soft_label
        # 标签由分数确定
        self.label = LABELS[np.argmax(soft_label)]
        
@DATASETS.register_module()
class HandSlideDataset(BaseDataset):
    def __init__(self,
                 data_prefix,
                 pipeline,
                 duration,
                 num_keypoint,
                 test_mode):
        self.duration = duration
        self.frames = parse_raw_files_to_frames(data_prefix, num_keypoint)
        self.actions = self.combine_actions()
        self.samples = []
        self.embedder = FullBodyPoseEmbedder()

        cache_path = os.path.join(data_prefix, "dataset_samples.pkl")
        if os.path.exists(cache_path):
            cache = torch.load(cache_path)
            self.samples = cache["samples"]
            print(f"加载samples缓存 {cache_path} ：生成时间 {cache['modify_time']}")
            # for s in self.samples:
                # s["img"] = s["embedding"]
                # s["gt_label"] = s["target"]
        else:
            for action in tqdm(self.actions):
                self.samples.append(self.convert_action_to_sample(action))
            print(f"生成缓存到 {cache_path}")
            torch.save({"samples": self.samples,
                        "modify_time": datetime.datetime.now()},
                       os.path.join(data_prefix, "dataset_samples.pkl"))
        
        # 统计样本分布
        sample_statics = defaultdict(int)
        for e in self.samples:
            sample_statics[e["label"]] += 1
        
        print("样本统计：")
        print(sample_statics)
        
        if not test_mode:
            none_samples  =[s for s in self.samples if s["label"] == "none"]
            down_samples  =[s for s in self.samples if s["label"] == "down"]
            up_samples  =[s for s in self.samples if s["label"] == "up"]
            sampled_nones = random.sample(none_samples, sample_statics["down"] + sample_statics["up"])
            print(f"滤除不平衡none样本, 保留 {len(sampled_nones)} 个none")
            self.samples = down_samples + up_samples + none_samples
        # 必须在完成samples的全部处理之后再构造父类，否则加载的samples可能是不完善的
        super().__init__(data_prefix, pipeline)
                    
    def load_annotations(self):
        data_infos = []
        data_infos = self.samples
        return data_infos
        

    # def __len__(self) -> int:
    #     return len(self.samples)

    # def __getitem__(self, index):
    #     return self.samples[index]
    
    def combine_actions(self):
        samples = []
        for i in range(len(self.frames) - self.duration + 1):
            sample = self.generate_single_action([i, i + self.duration - 1])
            samples.append(sample)
        return samples
    
    def convert_action_to_sample(self, action):
        assert isinstance(action, Action), type(action)
        result = {"landmark": ...,
            "label": action.label,
            "src_depth_paths": [f.depth_path for f in action.frames]}
        
        landmarks_np = np.array([frame.landmark for frame in action.frames])
        embedding_np = np.array([self.embedder(frame.landmark) for frame in action.frames])
        result["landmark"] = landmarks_np
        result["embedding"] = embedding_np
        # 向mmclassification的BaseClassifier低头
        result["img"] = embedding_np
        result["gt_label"] = action.soft_label

        return result
            
    
    def generate_single_action(self, indexes):
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
        # 2. 按照vap和当前片段的结束帧对齐程度判定类别和分数
        # best_score = 0
        # for k, vap in enumerate(valid_action_patches):
        #     # 计算vap和当前片段的tiof
        #     inter = max(0, min(vap[-1], indexes[-1]) - max(vap[0], indexes[0]) + 1)
        #     # union = max(vap[-1], indexes[-1]) - min(vap[0], indexes[0])
        #     tiof = inter / (indexes[-1] - indexes[0] + 1)
        #     # if tiof < 0.5:
        #         # continue
        #     score = 1 - abs(indexes[-1] - vap[-1]) / (vap[-1] - vap[0] + 1)
        #     if score >= best_score :#and score != 0:
        #         # 当前片段的分数
        #         best_score = score
        # action = Action(indexes[0], indexes[1],  [self.frames[k] for k in range(indexes[0], indexes[1] + 1)], best_score)
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
        action = Action(indexes[0], indexes[1],  [self.frames[k] for k in range(indexes[0], indexes[1] + 1)], soft_label)

        return action


