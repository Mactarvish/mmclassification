# author: muzhan
import time
import io
import json
import numpy as np
 
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
 
from tqdm import tqdm
import numpy as np
import cv2
import glob
import argparse
import os

from mmcls.datasets.pipelines.transforms import LandmarkNormalize
if __name__ == "__main__":
    from vis_infer_result import vis_depth_to_u8, vis_bbox, vis_kp, vis_on_table
else:
    from .vis_infer_result import vis_depth_to_u8, vis_bbox, vis_kp, vis_on_table


def create_gif(images_np, gif_name, duration_s=0.2):
    '''
    把images里的图像排成一个gif动态图，每帧的间隔是duration
    :param images: np组成的列表，例如[img1_np, img2_np, img3_np]
    :param gif_name:
    '''
    import imageio
    imageio.mimsave(gif_name, images_np, 'GIF', duration=duration_s)


def vis_hand_pose_3d(hand_poses_np, normalize=True):
    if len(hand_poses_np.shape) == 2:
        hand_poses_np = hand_poses_np[np.newaxis, ...]
    if normalize:
        ln = LandmarkNormalize()
        hand_poses_np = ln.normalize_landmark(hand_poses_np)
    
    vis_nps = []
    for hand_pose_np in hand_poses_np:
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
        # 画点
        ax.scatter3D(xp, yp, zp, cmap='Greens')
        # 连线
        if len(xp) == 16:
            for i in range(5):
                ax.plot(np.hstack((xp[i + 0], xp[i + 5], xp[i + 10], xp[15])),
                        np.hstack((yp[i + 0], yp[i + 5], yp[i + 10], yp[15])),
                        np.hstack((zp[i + 0], zp[i + 5], zp[i + 10], zp[15])),
                        ls='-', color=colors[i])
        elif len(xp) == 4:
            ax.plot(np.hstack((xp[0], xp[1], xp[2], xp[3])),
                    np.hstack((yp[0], yp[1], yp[2], yp[3])),
                    np.hstack((zp[0], zp[1], zp[2], zp[3])),
                    ls='-', color=colors[0])
        else:
            raise ValueError(len(xp))
        
        buf = io.BytesIO()
        plt.savefig(buf, format="jpg", dpi=90)
        # 必须关闭，否则越来越慢
        plt.close()
        buf.seek(0)
        img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
        buf.close()
        
        img = cv2.imdecode(img_arr, 1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        vis_nps.append(img)
    return vis_nps


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("src_json_dir")
    parser.add_argument("--single_finger", action="store_true")
    args = parser.parse_args()

    src_json_paths = glob.glob(os.path.join(args.src_json_dir, "**", "*.json"), recursive=True)

    for src_json_path in tqdm(src_json_paths):
        print(src_json_path)
        src_depth_path = src_json_path.replace("merge_result", "depth").replace(".json", ".png")
        assert os.path.exists(src_depth_path), src_depth_path

        json_dict = json.load(open(src_json_path, 'r'))
        if "kp" in json_dict:
            hand_pose_np = np.array(json_dict["kp"])[:, :3][np.newaxis, ...]
            vis_np = vis_hand_pose_3d(hand_pose_np, single_finger=args.single_finger, normalize=True)
            cv2.imwrite("ee.png", vis_np)
