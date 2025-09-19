import sys
import os
from pathlib import Path
from .Transformation import transform, save
from ..Distribution.Distribution import get_data


def equilibrate_data(data, path):
    best_v = max(data.values())
    values = list(data.values())
    paths = list(data.keys())
    relaunch = False
    for i in range(len(values)):
        if values[i] == best_v:
            continue
        new_path = os.path.join(path, paths[i])
        save_dir = os.path.abspath(new_path)
        imgs = os.listdir(new_path)
        img = 0
        while (values[i] != best_v):
            if img == len(imgs):
                break
            img_path = os.path.join(new_path, imgs[img])
            t_imgs = []
            if best_v - values[i] >= 6:
                t_imgs = [transform(img_path, j) for j in range(6)]
            else:
                t_imgs = [
                    transform(img_path, j) for j in range(best_v - values[i])
                    ]
            save(t_imgs, save_dir, img_path, False)
            values = list(get_data(path).values())
            if values[i] == best_v:
                break
            print(f"Number: {values[i]} | Goal: {best_v}")
            img += 1
            if img == len(imgs):
                relaunch = True
                break
    if relaunch:
        equilibrate_data(get_data(path), path)


def main():
    if len(sys.argv) != 2:
        print("Path folder is missing.")
        exit(1)
    pathname = Path(sys.argv[1])
    data = get_data(pathname)
    equilibrate_data(data, pathname)


if __name__ == "__main__":
    main()
