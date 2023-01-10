import sys
import os
import argparse
import re
import cv2


def generate_images_html(image_paths, scale_factor=1.):
    if not isinstance(image_paths, list):
        image_paths = [image_paths]
    image_sizes = []

    tds = ["<br><img src=\"%s\"  border=1 /> <br>" % p for p in image_paths]

    begin = '''<meta http-equiv="Content-Type" content="text/html; charset=utf-8" />
 <table style="word-break:break-all; word-wrap:break-all;"><tr>'''
    end = '''</tr></table>'''
    ths = ["<th>%s</th>" % p for p in image_paths]
    return begin + '\n' + '\n'.join(ths + tds) + '\n' + end


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dst_dir")
    parser.add_argument("--sf", default=1., type=float)
    parser.add_argument("--max_show", default=-1., type=int)
    parser.add_argument("--shuffle", action="store_true")
    args = parser.parse_args()
    dst_dir = args.dst_dir.rstrip('/')

    if not dst_dir.startswith('/'):
        dst_dir = os.path.abspath(dst_dir)
    file_names = list(filter(lambda n: any([n.endswith(p) for p in [".jpg", ".png", ".gif"]]), os.listdir(dst_dir)))
    if args.shuffle:
        import random
        import time
        random.seed(time.time())
        random.shuffle(file_names)
    print("Total %d files under %s\n" % (len(file_names), dst_dir))

    if args.max_show != -1:
        file_names = file_names[:args.max_show]
    if len(file_names) == 0:
        print("No file found, abort.")
        exit(0)
    file_folder_name = os.path.split(dst_dir)[-1]
    file_paths = sorted([os.path.join(dst_dir, n) for n in file_names])
    html_str = generate_images_html(file_paths, args.sf)
    html_save_path = os.path.join("/tmp", "%s.html" % os.path.split(dst_dir)[-1])

    with open(html_save_path, 'w') as f:
        f.write(html_str)
    print("html save path: %s" % "http://localhost:8000/tmp/%s.html" % os.path.split(dst_dir)[-1])