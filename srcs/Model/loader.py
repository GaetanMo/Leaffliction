from pathlib import Path
from typing import Dict, Tuple
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


def default_transform(img_size: int):
    """Apply the same preprocessing as training/validation to a single PIL image.

    This mirrors the Resize->CenterCrop->ToTensor->Normalize pipeline used in loaders.
    """
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    tfm = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    return tfm


def build_loaders(
    data_dir: str | Path = "leaves/images",
    img_size: int = 224,
    batch_size: int = 32,
    val_split: float = 0.2,
    seed: int = 42,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader, Dict[str, int]]:

    train_tfms = default_transform(img_size)
    val_tfms = default_transform(img_size)

    # Load once; we'll clone with different transforms
    base = datasets.ImageFolder(str(data_dir))
    class_to_idx = base.class_to_idx

    # Deterministic split
    n_total = len(base)
    n_val = int(n_total * val_split)
    n_train = n_total - n_val
    gen = torch.Generator().manual_seed(seed)
    train_subset, val_subset = random_split(base, [n_train, n_val], generator=gen)

    # Attach transforms per split
    train_subset.dataset = datasets.ImageFolder(str(data_dir), transform=train_tfms)
    val_subset.dataset = datasets.ImageFolder(str(data_dir), transform=val_tfms)

    train_loader = DataLoader(
        train_subset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader, class_to_idx
