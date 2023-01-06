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

from .base_dataset import BaseDataset
from .builder import DATASETS

LABELS = ["none", "up", "down", "left", "right"][:3]
if len(LABELS) == 3:
    print("不考虑左右")
    print("不考虑左右")
    print("不考虑左右")
    print("不考虑左右")
    print("不考虑左右")
    print("不考虑左右") 


@DATASETS.register_module()
class HandSlideDatasetRGB(BaseDataset):
    def __init__(self,
                 src_dir,
                 pipeline,
                 duration,
                 num_keypoints,
                 single_finger,
                 test_mode,
                 gt_per_frame):
        # assert "backup" not in src_dir, src_dir
        # assert "slide/" in src_dir, src_dir
        self.src_dir = src_dir
        
        self.duration = duration
        self.gt_per_frame = gt_per_frame
        self.single_finger = single_finger
        self.frames = self.parse_jsons_to_frames(self.src_dir, num_keypoints)

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
        
        if self.single_finger:
            print("使能单指推理")
            for s in self.samples:
                # 保留 [1,6,11,16]
                s["img"][:, 0, :] = 0
                s["img"][:, 2:6, :] = 0
                s["img"][:, 7:11, :] = 0
                s["img"][:, 12:16, :] = 0

        # 统计样本分布
        self.sample_statics = defaultdict(int)
        for e in self.samples:
            self.sample_statics[e["label"]] += 1
        
        print("样本统计：")
        print(self.sample_statics)
        
        if True:
            none_samples  =[s for s in self.samples if s["label"] == "none"]
            down_samples  =[s for s in self.samples if s["label"] == "down"]
            up_samples  =[s for s in self.samples if s["label"] == "up"]
            sampled_nones = random.sample(none_samples, self.sample_statics["down"] + self.sample_statics["up"])
            print(f"滤除不平衡none样本, 保留 {len(sampled_nones)} 个none")
            self.samples = down_samples + up_samples + sampled_nones


        # 统计样本分布
        self.sample_statics = defaultdict(int)
        for e in self.samples:
            self.sample_statics[e["label"]] += 1
        
        print("样本统计：")
        print(self.sample_statics)


        if self.gt_per_frame:
            print("使能逐帧GT")
            for s in self.samples:
                s["gt_label"] = s["frame_label"]
        super().__init__(self.src_dir, pipeline)
                    
    def parse_jsons_to_frames(self, src_dir, num_keypoints):
        src_json_paths = sorted(glob.glob(os.path.join(self.src_dir, "**", "merge_result", "*.json"), recursive=True))
        assert len(src_json_paths) != 0, self.src_dir
        # 首先把没有框的样本剔除，保持后续遍历时索引的连续性
        t = []
        for src_json_path in src_json_paths:
            jd = json.load(open(src_json_path, 'r'))
            if "kp" in jd and "on_table" in jd:
                t.append(src_json_path)
        src_json_paths = t
        print(len(src_json_paths))
        # src_json_paths = list(filter(lambda p: "kp" in json.load(open(p, 'r')), src_json_paths))
        
        # 逐个json解析成Frame
        frames = []
        for i, src_json_path in enumerate(src_json_paths):
            json_dict = json.load(open(src_json_path, 'r'))
            # assert "bbox" in json_dict
            assert "kp" in json_dict
            assert "on_table" in json_dict

            # 解析landmark
            landmark = np.zeros((num_keypoints , 2), dtype=np.float32)
            for j in range(num_keypoints):
                landmark[j] = json_dict["kp"][j][:2]
                
            # 确定当前帧的标签
            if json_dict["on_table"]:
                for e in LABELS:
                    # if '/' + e in src_json_path:
                    if e in src_json_path:
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
        if self.gt_per_frame:
            # 除了0以外，最多且超过2个的元素
            gt_labels = []
            for e in self.data_infos:
                class_count = e["frame_label"].argmax(axis=1)
                maxc = -1
                maxi = 0
                for i in range(self.data_infos[0]["frame_label"].shape[-1]):
                    if (class_count == i).sum() > maxc:
                        maxc = (class_count == i).sum()
                        maxi = i
                if maxc > 2:
                    gt_labels.append(maxi)
                else:
                    gt_labels.append(0)
            gt_labels = np.array(gt_labels)
        else:
            gt_labels = np.array([data['gt_label'] for data in self.data_infos]).argmax(axis=1)
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

        frames = [self.frames[k] for k in range(indexes[0], indexes[1] + 1)]
        result = dict()
        result["src_depth_paths"]  =[f.depth_path for f in frames]
        # result["img"] = np.array([self.normalize_landmark(f.landmark) for f in frames])
        result["img"] = np.array([f.embedding for f in frames])
        result["gt_label"] = soft_label
        result["label"] = LABELS[np.argmax(soft_label)]
        result["frame_label"] = [LABELS.index(f.label) for f in frames]
        # 转one hot编码
        result["frame_label"] = np.eye(len(LABELS))[result["frame_label"]]

        return result

    def __str__(self):
        dataset_name = os.path.basename(self.src_dir)
        s = ' '.join([dataset_name, self.sample_statics.__str__()])
        return s


class Frame():
    def __init__(self, labelme_path, landmark, label, num_frame) -> None:
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
        