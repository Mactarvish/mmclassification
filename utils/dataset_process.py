import shutil
from tqdm import tqdm
import random
from tabulate import tabulate
import json
from ma_dl_kit.utils.glob_pattern import glob_json_paths
from collections import defaultdict
import os

import glob

def split_name_updown_to_name_contain_updown(root_dir):
    src_dirs = glob.glob(os.path.join(root_dir, '*'))
    # print(src_dirs)
    for src_dir in src_dirs:
        bn = os.path.basename(src_dir)
        date_dir = os.path.dirname(src_dir)
        name, label = bn.split('_')
        # print(name, label)
        name_dir = os.path.join(date_dir, name)
        os.makedirs(name_dir, exist_ok=True)
        # print(name_dir)
        new_dir = os.path.join(name_dir, label)
        print(src_dir)
        print(new_dir)
        os.rename(src_dir, new_dir)
        print()

def add_front_frame_on_table(root_dir):
    src_json_paths = glob.glob(os.path.join(root_dir, "**", "*.json"), recursive=True)
    src_json_dirs = set(os.path.dirname(p) for p in src_json_paths)
    # print(src_dirs)
    [print(p) for p in src_json_dirs]
    
    for src_json_dir in tqdm(src_json_dirs):
        src_json_paths = sorted(glob.glob(os.path.join(src_json_dir, "*.json")))
        to_be_added_json_paths = []
        # [print(p) for p in src_json_paths]
        for i in range(1, len(src_json_paths)):
            cur_jd = json.load(open(src_json_paths[i]))
            last_jd = json.load(open(src_json_paths[i - 1]))

            if cur_jd["on_table"] and not last_jd["on_table"]:
                to_be_added_json_paths.append(src_json_paths[i - 1])
        # print(to_be_added_json_paths)
        # [print(p) for p in to_be_added_json_paths]
        for jp in to_be_added_json_paths:
            jd = json.load(open(jp))
            jd["on_table"] = True
            json.dump(jd, open(jp, 'w'))



if __name__ == '__main____':
    '''
    |  类别  | 数量   | 平均时长   |
|:------:|:-------|:-----------|
|  none  | 3965   | 3.14401    |
|   up   | 707    | 2.85007    |
|  down  | 1271   | 3.23839    |
| 类别   | Precision   | Recall   | F1-Score   |
|:-------|:------------|:---------|:-----------|
| 0      | 97.417      | 84.641   | 90.58      |
| 1      | 60.284      | 96.181   | 74.114     |
| 2      | 88.248      | 95.122   | 91.556     |'''

    gt_0 = 408
    gt_1 = 403
    pred_0_from_gt_0 = 281
    pred_1_from_gt_0 = gt_0 - pred_0_from_gt_0
    pred_0_from_gt_1 = 146
    pred_1_from_gt_1 = gt_1 - pred_0_from_gt_1

    precision_0 = pred_0_from_gt_0 / (pred_0_from_gt_0 + pred_0_from_gt_1)
    recall_0 = pred_0_from_gt_0 / gt_0
    f1_0 = 2 * (precision_0 * recall_0) / (precision_0 + recall_0)


    precision_1 = pred_1_from_gt_1 / (pred_1_from_gt_1 + pred_1_from_gt_0)
    recall_1 = pred_1_from_gt_1 / gt_1
    f1_1 = 2 * (precision_1 * recall_1) / (precision_1 + recall_1)

    table_data = []
    table_data = [["down_input"] + ["%.3f" % e for e in [precision_0, recall_0, f1_0]], 
                  ["up_return"] + ["%.3f" % e for e in [precision_1, recall_1, f1_1]]]
    # 类别
    table = tabulate(
    table_data,
    headers=["类别", "Precision", "Recall", "F1-Score"],
    tablefmt="pipe",
    numalign="left",
    stralign="center",
    )

    
    print(table)