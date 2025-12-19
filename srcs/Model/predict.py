import argparse
from pathlib import Path
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from .loader import default_transform
from .model import build_model


def load_image(path: Path, img_size: int) -> torch.Tensor:
    with Image.open(path) as img:
        img = img.convert("RGB")
        tfm = default_transform(img_size)
        t = tfm(img)
    return t.unsqueeze(0)


def show_prediction_window(pred_img_path: Path, orig_img_path: Path, pred_class: str, conf: float) -> None:
    """Display the prediction alongside paths to the transformed and original images."""
    with Image.open(orig_img_path).convert("RGB") as img_orig:
        np_orig = np.asarray(img_orig)
    with Image.open(pred_img_path).convert("RGB") as img_pred:
        np_pred = np.asarray(img_pred)

    fig = plt.figure(
        num="Leaf Prediction",
        figsize=(9, 7.5),
        facecolor="#222222",
        constrained_layout=True,
    )
    gs = fig.add_gridspec(3, 2, height_ratios=[5, 0.8, 1.1])
    gs.update(hspace=0.04)

    ax_orig = fig.add_subplot(gs[0, 0])
    ax_pred = fig.add_subplot(gs[0, 1])
    ax_title = fig.add_subplot(gs[1, :])
    ax_text = fig.add_subplot(gs[2, :])

    ax_orig.imshow(np_orig)
    ax_pred.imshow(np_pred)
    for ax in (ax_orig, ax_pred):
        ax.axis("off")
        ax.set_facecolor("#222222")

    ax_title.set_facecolor("#222222")
    ax_title.axis("off")
    ax_title.text(
        0.5,
        0.5,
        "DL classification",
        color="white",
        ha="center",
        va="center",
        fontsize=20,
        fontweight="bold",
    )

    ax_text.set_facecolor("#222222")
    ax_text.axis("off")
    ax_text.text(
        0.5,
        0.75,
        f"Class predicted : {pred_class}",
        color="#7ed957",
        ha="center",
        va="center",
        fontsize=16,
        fontweight="bold",
    )

    plt.show(block=True)


def get_args():
    parser = argparse.ArgumentParser(description="Predict a class for a single image")
    parser.add_argument("img_path", type=str, help="Path to the transformed image used for prediction")
    parser.add_argument("orig_path", type=str, help="Path to the original image (for display only)")
    parser.add_argument("--checkpoint", type=str, default="Model/checkpoints/best.pt")
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--show", action="store_true", help="Display the image with the predicted label overlay")
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    checkpoint = Path(args.checkpoint)
    img_path = Path(args.img_path)
    orig_img_path = Path(args.orig_path)
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    if not img_path.exists():
        raise FileNotFoundError(f"Image not found: {img_path}")
    if not orig_img_path.exists():
        raise FileNotFoundError(f"Original image not found: {orig_img_path}")
    exts = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
    if not img_path.is_file() or img_path.suffix not in exts:
        raise ValueError(f"File {img_path} is not a supported image type")
    if not orig_img_path.is_file() or orig_img_path.suffix not in exts:
        raise ValueError(f"File {orig_img_path} is not a supported image type")

    checkpoint_loaded = torch.load(checkpoint, map_location="cpu")
    class_to_idx = checkpoint_loaded.get("class_to_idx")
    if not class_to_idx:
        raise ValueError("Checkpoint missing 'class_to_idx'")
    model = build_model(len(class_to_idx))
    model.load_state_dict(checkpoint_loaded.get("model_state"), strict=False)
    model.eval()

    x = load_image(img_path, args.img_size)
    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=1)[0]
        conf, pred_idx = torch.max(probs, dim=0)

    idx_to_class = {v: k for k, v in class_to_idx.items()}
    pred_class = idx_to_class[int(pred_idx)]
    print(f"Prediction: {pred_class} (p={float(conf):.4f})")
    if args.show:
        show_prediction_window(img_path, orig_img_path, pred_class, float(conf))


if __name__ == "__main__":
    main()
