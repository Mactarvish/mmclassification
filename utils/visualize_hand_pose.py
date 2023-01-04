# author: muzhan
import matplotlib
import json
import numpy as np
 
matplotlib.use('Agg')
import matplotlib.pyplot as plt
 
 
from tqdm import tqdm
import numpy as np
import cv2
import glob
import argparse
import os

from mmcls.datasets.pipelines.transforms import LandmarkNormalize
from vis_infer_result import vis_depth_to_u8, vis_bbox, vis_kp, vis_on_table


def vis_hand_pose_3d(hand_pose_np, single_finger=False):
    assert hand_pose_np.shape == (16, 3)
    # 交换xy，这样可视化更清晰
    yp = hand_pose_np[:, 0]
    xp = hand_pose_np[:, 1]
    zp = 1- hand_pose_np[:, 2]

    ax = plt.axes(projection='3d')
    ax.set_xlim3d([-0.5, 0.5])
    ax.set_ylim3d([0, 1.])
    ax.set_zlim3d([0.5, 1.])
    ax.view_init(elev=15., azim=70)
    ax.dist = 7
    
    # right leg
    colors = ["blue", "green", "yellow", "red", "cyan"]
    if single_finger:
        i = 1
        ax.scatter3D(xp[[1,6,11,15]], yp[[1,6,11,15]], zp[[1,6,11,15]], cmap='Greens')
        ax.plot(np.hstack((xp[i + 0], xp[i + 5], xp[i + 10], xp[15])),
                np.hstack((yp[i + 0], yp[i + 5], yp[i + 10], yp[15])),
                np.hstack((zp[i + 0], zp[i + 5], zp[i + 10], zp[15])),
                ls='-', color=colors[i])
    else:
        # 画点
        ax.scatter3D(xp, yp, zp, cmap='Greens')
        # 连线
        for i in range(5):
            ax.plot(np.hstack((xp[i + 0], xp[i + 5], xp[i + 10], xp[15])),
                    np.hstack((yp[i + 0], yp[i + 5], yp[i + 10], yp[15])),
                    np.hstack((zp[i + 0], zp[i + 5], zp[i + 10], zp[15])),
                    ls='-', color=colors[i])
    
    plt.savefig('skeleton.png')



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("src_json_dir")
    parser.add_argument("--single_finger", action="store_true")
    args = parser.parse_args()

    src_json_paths = glob.glob(os.path.join(args.src_json_dir, "**", "*.json"), recursive=True)

    ln = LandmarkNormalize()
    for src_json_path in tqdm(src_json_paths):
        print(src_json_path)
        src_depth_path = src_json_path.replace("merge_result", "depth").replace(".json", ".png")
        assert os.path.exists(src_depth_path), src_depth_path

        json_dict = json.load(open(src_json_path, 'r'))
        if "kp" in json_dict:
            hand_pose_np = np.array(json_dict["kp"])[:, :3][np.newaxis, ...]
            hand_pose_np = ln.normalize_landmark(hand_pose_np)[0]
            vis_hand_pose_3d(hand_pose_np, single_finger=args.single_finger)
