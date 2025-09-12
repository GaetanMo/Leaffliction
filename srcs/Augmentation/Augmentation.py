from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import argparse
from pathlib import Path
import albumentations
import numpy as np
import os


def is_valid(path):
    path = Path(path)
    if not path.exists():
        print("Path does not exists.")
        return 0
    try:
        with Image.open(path) as img:
            if img.format == "JPEG":
                return 1
            else:
                print("Invalid format.")
                return 0
    except Exception as e:
        print("File is not an image.")
        return 0

def transform(path, i):
    img = np.array(Image.open(path))
    transforms = [
        albumentations.Rotate(limit=45, p=1),
        albumentations.Blur(blur_limit=7, p=1),
        albumentations.RandomBrightnessContrast(brightness_limit=0, contrast_limit=0.5, p=1),
        albumentations.RandomCrop(width=200, height=200, p=1),
        albumentations.RandomBrightnessContrast(brightness_limit=0.5, contrast_limit=0, p=1),
        albumentations.Affine(shear=20, p=1)
    ]
    aug = albumentations.Compose([transforms[i]])
    augmented = aug(image=img)
    return augmented['image']

def save(imgs, dir, original_path):
    if not os.path.isdir(dir):
        print(f"'{dir}' is not a directory.")
        exit(1)
    if not os.access(dir, os.W_OK):
        print(f"Can't write in '{dir}'.")
        exit(1)
    original_name = os.path.splitext(os.path.basename(original_path))[0]
    transformations = [
        "Original",
        "Rotate",
        "Blur",
        "Contrast",
        "Crop",
        "Illumination",
        "Affine"
    ]
    for i, img in enumerate(imgs):
        if i == 0:
            continue
        path = os.path.join(dir, f"{original_name}_{transformations[i]}.JPG")
        img_pil = Image.fromarray(img)
        img_pil.save(path)
    print(f"Images succesfully saved in {dir}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="Chemin du fichier à traiter")
    parser.add_argument("-d", "--display", choices=["on", "off"], default="on", help="Display image transformations (on/off, default: on)")
    parser.add_argument("-s", "--save", metavar="directory", help="Save image transformations in the directory.")
    args = parser.parse_args()
    if not is_valid(args.path):
        exit(1)
    img_original = np.array(Image.open(args.path))
    images = [img_original] + [transform(args.path, i) for i in range(6)]
    save(images, args.save, args.path)
    if args.display == "on":
        fig, axes = plt.subplots(1, 7, figsize=(18, 4))
        for ax, img in zip(axes, images):
            ax.imshow(img)
            ax.axis("off")
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()