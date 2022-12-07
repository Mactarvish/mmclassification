import argparse
from collections import defaultdict
import os
import re
import glob
import time
import cv2
import json
import numpy as np
from tqdm import tqdm

from .poseembedding import FullBodyPoseEmbedder

LABELS = ["none", "up", "down"]

STANDARD_ACTION_SIZE = 5


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



    
    
def parse_single_frame(labelme_path, label, num_keypoint) -> Frame:
    src_depth_path = labelme_path.replace(".json", ".png").replace("normal", "depth")
    src_depth_np = cv2.imread(src_depth_path, cv2.IMREAD_UNCHANGED)
    landmark = np.zeros((num_keypoint, 3), dtype=np.float32)
    # print(src_depth_path)

    json_dict = json.load(open(labelme_path, 'r'))
    for item in json_dict["shapes"]:
        if item["label"] == "finger1":
            x, y = item["points"][0]
            z = src_depth_np[round(y), round(x)]
            landmark[0] = [x, y, z]
        elif item["label"] == "finger2":
            x, y = item["points"][0]
            z = src_depth_np[round(y), round(x)]
            landmark[1] = [x, y, z]

    frame = Frame(labelme_path, landmark, label)
    
    return frame

def parse_on_table(src_dir):
    # 每个目录里只有一种动作
    assert ("/down" in src_dir) ^ ("/up" in src_dir), src_dir
    if "down" in src_dir:
        action_name = "down"
    else:
        action_name = "up"

    action_json_dir = os.path.join(src_dir, "infer_result")
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
    
    # 组合Action
    actions = []
    
    sizes = defaultdict(list)
    for action_patch in action_patches:
        begin, end, action = action_patch
        size = end - begin + 1
        sizes[action].append(size)

    print("动作帧数统计：")
    print(sizes)
    for key in sizes:
        values = set(sizes[key])
        print(key, {v: sizes[key].count(v) for v in values})
         
    for i in range(STANDARD_ACTION_SIZE - 1, len(frames)):
        
        Action(i - STANDARD_ACTION_SIZE, i, xx, [frames[k] for k in range(i - STANDARD_ACTION_SIZE, i + 1)], xx)
    
    for action_patch in action_patches:
        begin, end, action = action_patch
        size = end - begin + 1
        num_frames = []
        if size != STANDARD_ACTION_SIZE:
            print(f"注意：动作时长 {size} 不等于标准时长", end='，')
            if end < STANDARD_ACTION_SIZE:
                print("结束帧都tm小于标准帧数，用第一帧补齐")
                num_frames = (STANDARD_ACTION_SIZE - end) * [1] + list(range(begin, end))
            else:
                if size < STANDARD_ACTION_SIZE:
                    print("前向补齐，用前序动作帧补齐标准帧数")
                else:
                    print("前向截断，只保留最后的标准帧数")
                begin = end - STANDARD_ACTION_SIZE + 1
                num_frames = list(range(begin, end + 1))
        else:
                num_frames = list(range(begin, end + 1))
        action = Action(begin, end, action, [frames[i] for i in num_frames], 1)
        actions.append(action)
        action.generate_sample()
    
    print(len(actions))

    return actions, frames

    # 可视化检查
    vis_save_dir = "pgs/vis"
    os.makedirs(vis_save_dir, exist_ok=True)
    for i in tqdm(range(min(200, len(frames)))):
        depth_np = cv2.imread(frames[i].depth_path, cv2.IMREAD_UNCHANGED)
        src_image_name = os.path.basename(frames[i].depth_path)
        vis_np = vis_depth_to_u8(depth_np)
        cv2.putText(vis_np, frames[i].label, (10, 400), cv2.FONT_HERSHEY_COMPLEX, 3, (0, 255, 0))
        cv2.imwrite(os.path.join(vis_save_dir, src_image_name), vis_np)


    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Python")
    parser.add_argument("src_dir", type=str)
    parser.add_argument("--num_keypoint", type=int, default=21)
    args = parser.parse_args()

    parse_raw_files_to_actions(args)
    
    

