from plantcv import plantcv as pcv
import cv2
import numpy as np
import os
import argparse
import matplotlib.pyplot as plt

def transformations(img, mask_option=False):
    # Get gaussian blur image
    blur = cv2.GaussianBlur(img, (7, 7), 0)

    # Get mask
    # Pixels supérieurs à 100 → objet, pixels inférieurs → fond
    mask = pcv.threshold.binary(img, threshold=100, object_type='light')

    # Get ROI
    height, width = img.shape[:2]
    x, y, w, h = 0, 0, width, height
    rect_roi = pcv.roi.rectangle(img, x, y, w, h)
    a_gray = pcv.rgb2gray_lab(rgb_img=img, channel="a")
    bin_mask = pcv.threshold.otsu(gray_img=a_gray, object_type="dark")
    cleaned_mask = pcv.fill(bin_img=bin_mask, size=50)
    filtered_mask  = pcv.roi.filter(mask=cleaned_mask, roi=rect_roi, roi_type='partial')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    green_layer = np.zeros_like(img)
    green_layer[:, :, 1] = filtered_mask
    output = gray_bgr.copy()
    output[filtered_mask > 0] = green_layer[filtered_mask > 0]
    cv2.rectangle(output, (x, y), (x + w, y + h), color=(255, 0, 0), thickness=3)  # BGR -> bleu

    # Get object analysis
    shape_img = pcv.analyze.size(img=img, labeled_mask=filtered_mask)

    # PSEUDOLANDMARKS A FAIRE

    return [("blur", blur), ("mask", mask), ("roi", output), ("shape_analysis", shape_img)]

def process_image(path, dst_dir=None, mask_option=False):
    img, path, filename = pcv.readimage(path)
    if img is None:
        print(f"Error : impossible to read {path}")
        return

    transformed = transformations(img, mask_option)

    for name, t_img in transformed:
        if dst_dir:
            os.makedirs(dst_dir, exist_ok=True)
            base_name = os.path.splitext(os.path.basename(path))[0]
            save_path = os.path.join(dst_dir, f"{base_name}_{name}.JPG")
            cv2.imwrite(save_path, t_img)
        else:
            plt.imshow(cv2.cvtColor(t_img, cv2.COLOR_BGR2RGB))
            plt.title(name)
            plt.axis('off')
            plt.show()

def main():
    parser = argparse.ArgumentParser(description="Leaf image transformations")
    parser.add_argument("-src", required=True, help="Source image or folder")
    parser.add_argument("-dst", help="Destination folder to save results")
    parser.add_argument("-mask", action="store_true", help="Display the colored mask")
    args = parser.parse_args()

    if os.path.isdir(args.src):
        # Directory
        if args.dst is None:
            print("Error: -dst is required when -src is a directory.")
            return
        for filename in os.listdir(args.src):
            if filename.endswith((".JPG")):
                process_image(os.path.join(args.src, filename), dst_dir=args.dst, mask_option=args.mask)
    else:
        # File
        process_image(args.src, dst_dir=args.dst, mask_option=args.mask)

if __name__ == "__main__":
    main()
