import random
import  glob
import os
from tqdm import tqdm
import shutil
import json
import cv2
import numpy as np
import sys
import argparse


def vis_depth_to_u8(src_depth_path):
    if isinstance(src_depth_path, str):
        assert os.path.exists(src_depth_path), src_depth_path
        src_np = cv2.imread(src_depth_path, cv2.IMREAD_UNCHANGED)
    elif isinstance(src_depth_path, np.ndarray):
        src_np = src_depth_path
    if len(src_np.shape) == 2:
        u8_np = np.uint8(src_np.astype(np.float32) / 1200 * 255)[..., np.newaxis].repeat(3, 2)
    else:
        u8_np = np.uint8(src_np.astype(np.float32) / 1200 * 255)
    return u8_np



def vis_bbox(vis_np, json_dict):
    if "bbox" not in json_dict:
        return vis_np
    x1,y1,x2, y2 = json_dict["bbox"][1:-1]
    cv2.rectangle(vis_np, (x1, y1), (x2, y2), (0, 0, 255))
    return vis_np


def vis_kp(vis_np, json_dict):
    if "kp" not in json_dict:
        return vis_np
    kps_xy = [(int(e[0]), int(e[1])) for e in json_dict["kp"]]
    
    # mediapipe
    # for i in [1,5,9,13,17]:
    #     cv2.line(vis_np, kps_xy[i], kps_xy[i + 1], (0, 255, 255), 2)
    #     cv2.line(vis_np, kps_xy[i + 1], kps_xy[i + 2], (0, 255, 255), 2)
    #     cv2.line(vis_np, kps_xy[i + 2], kps_xy[i + 3], (0, 255, 255), 2)
    #     cv2.line(vis_np, kps_xy[i], kps_xy[0], (0, 255, 255), 2)

    for i in range(5):
        cv2.line(vis_np, kps_xy[i], kps_xy[i + 5], (0, 255, 255), 1)
        cv2.line(vis_np, kps_xy[i + 5], kps_xy[i + 10], (0, 255, 255), 1)
        cv2.line(vis_np, kps_xy[i + 10], kps_xy[15], (0, 255, 255), 1)
        
    
    for i in range(len(kps_xy)):
        cv2.circle(vis_np, kps_xy[i], 2, (255, 255, 0))
        
    return vis_np


def vis_on_table(vis_np, json_dict):
    if "on_table" in json_dict and json_dict["on_table"]:
        cv2.putText(vis_np, "on_table", (100, 100), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0))
    return vis_np



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("src_dir")
    parser.add_argument("--num_modals", type=int, default=4)
    args = parser.parse_args()
    
    infer_json_paths = glob.glob(os.path.join(args.src_dir, "**", "merge_result", "*.json"), recursive=True)

    for infer_json_path in tqdm(infer_json_paths):
        print(infer_json_path)
        src_depth_path = infer_json_path.replace("merge_result", "depth").replace(".json", ".png")
        assert os.path.exists(src_depth_path), src_depth_path
        json_dict = json.load(open(infer_json_path, 'r'))

        depth_np = vis_depth_to_u8(src_depth_path)
        vis_np = depth_np.copy()

        vis_np = vis_bbox(vis_np, json_dict)
        vis_np = vis_kp(vis_np, json_dict)
        vis_np = vis_on_table(vis_np, json_dict)
        
        vis_save_path = src_depth_path.replace("depth", "normal")
        os.makedirs(os.path.dirname(vis_save_path), exist_ok=True)
        if cv2.imwrite(vis_save_path, vis_np):
            print(vis_save_path)
            


