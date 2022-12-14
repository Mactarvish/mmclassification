import glob
import os



if __name__ == '__main__':
    cache_pkl_paths = glob.glob(os.path.join("/data/dataset/hand/backup/slide/", "**", "*.pkl"), recursive=True)
    for pp in cache_pkl_paths:
        print(pp)
        os.remove(pp)