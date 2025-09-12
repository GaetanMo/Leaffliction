import argparse
from pathlib import Path
import torch
import torch.nn.functional as F
from PIL import Image
from .loader import default_transform
from .model import build_model


def load_image(path: Path, img_size: int) -> torch.Tensor:
    with Image.open(path) as img:
        img = img.convert("RGB")
        tfm = default_transform(img_size)
        t = tfm(img)
    return t.unsqueeze(0)


def get_args():
    parser = argparse.ArgumentParser(description="Predict a class for a single image")
    parser.add_argument("img_path", type=str)
    parser.add_argument("--checkpoint", type=str, default="Model/checkpoints/best.pt")
    parser.add_argument("--img_size", type=int, default=224)
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    checkpoint = Path(args.checkpoint)
    img_path = Path(args.img_path)
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    if not img_path.exists():
        raise FileNotFoundError(f"Image not found: {img_path}")
    exts = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
    if not img_path.is_file() or img_path.suffix not in exts:
        raise ValueError(f"File {img_path} is not a supported image type")

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


if __name__ == "__main__":
    main()
