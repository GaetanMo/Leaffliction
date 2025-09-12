from pathlib import Path
from typing import Dict, Tuple
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


def default_transform(img_size: int):
    """Apply the same preprocessing as training/validation to a single PIL image.
    This mirrors the Resize->CenterCrop->ToTensor->Normalize pipeline used in by resnet (our model).
    """
    mean = (0.485, 0.456, 0.406)  # resnet (our model) requirement
    std = (0.229, 0.224, 0.225)  # resnet (our model) requirement
    tfm = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),  # Converts a PIL image (H×W×C, 0..255) to a float32 tensor C×H×W scaled to [0, 1]
            transforms.Normalize(mean, std),  # Puts inputs on the same scale the pretrained weights were trained on
        ]
    )
    return tfm


def build_loaders(
    data_dir: str | Path = "leaves/images",
    img_size: int = 224,
    batch_size: int = 32,
    val_split: float = 0.2,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, Dict[str, int]]:
    tfms = default_transform(img_size)
    # load
    base = datasets.ImageFolder(str(data_dir), transform=tfms)  # expects class_name/imagename.jpg layout
    class_to_idx = base.class_to_idx
    # split
    n_total = len(base)
    n_val = int(n_total * val_split)
    n_train = n_total - n_val
    gen = torch.Generator().manual_seed(seed)
    train_subset, val_subset = random_split(base, [n_train, n_val], generator=gen)
    # created DataLoader
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, pin_memory=True)
    return train_loader, val_loader, class_to_idx
