from pathlib import Path
import shutil
import random
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

def make_imagefolder_split(
    source_dir,
    output_dir,
    val_frac=0.2,
    seed=372,
    copy_files=True,
    ):
    """
    Create a saved train/val split from a class-folder dataset.

    source_dir:
        Root folder with one subfolder per class.
    output_dir:
        Root folder to create:
            output_dir/train/<class_name>/*.jpg
            output_dir/val/<class_name>/*.jpg
    val_frac:
        Fraction of each class to place in validation.
    seed:
        Random seed for reproducibility.
    copy_files:
        If True, copy files. If False, move files.
    """
    source_dir = Path(source_dir)
    output_dir = Path(output_dir)
    train_dir = output_dir / "train"
    val_dir = output_dir / "val"

    rng = random.Random(seed)

    if output_dir.exists():
        print(f"{output_dir} already exists, skipping split.")
        return

    train_dir.mkdir(parents=True, exist_ok=False)
    val_dir.mkdir(parents=True, exist_ok=False)

    class_dirs = [p for p in source_dir.iterdir() if p.is_dir()]
    class_dirs = sorted(class_dirs)

    total_train = 0
    total_val = 0

    for class_dir in class_dirs:
        class_name = class_dir.name
        image_paths = [p for p in class_dir.iterdir() if p.is_file()]
        image_paths = sorted(image_paths)

        if len(image_paths) < 2:
            print(f"Skipping class {class_name}: fewer than 2 images")
            continue

        rng.shuffle(image_paths)

        n_val = max(1, int(len(image_paths) * val_frac))
        n_val = min(n_val, len(image_paths) - 1)

        val_paths = image_paths[:n_val]
        train_paths = image_paths[n_val:]

        out_train_class = train_dir / class_name
        out_val_class = val_dir / class_name
        out_train_class.mkdir(parents=True, exist_ok=True)
        out_val_class.mkdir(parents=True, exist_ok=True)

        op = shutil.copy2 if copy_files else shutil.move

        for src in train_paths:
            op(src, out_train_class / src.name)
        for src in val_paths:
            op(src, out_val_class / src.name)

        total_train += len(train_paths)
        total_val += len(val_paths)

        print(
            f"{class_name}: train={len(train_paths)} val={len(val_paths)} total={len(image_paths)}"
        )

    print(f"\nDone.")
    print(f"Train images: {total_train}")
    print(f"Val images:   {total_val}")
    print(f"Saved split to: {output_dir}")


def get_dataloaders(
    split_root,
    batch_size=32,
    image_size=224,
    num_workers=4,
    use_augmentation=False,
):
    """
    Build train and validation dataloaders from a saved ImageFolder split.

    Expected structure:
        split_root/train/<class_name>/*
        split_root/val/<class_name>/*
    """
    split_root = Path(split_root)
    train_dir = split_root / "train"
    val_dir = split_root / "val"

    normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
    )


    if use_augmentation:
        train_tfms = transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.05),
            transforms.ToTensor(),
            normalize,\
        ])
    else:
        train_tfms = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            normalize,
        ])

    val_tfms = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        normalize,
    ])

    train_ds = datasets.ImageFolder(train_dir, transform=train_tfms)
    val_ds = datasets.ImageFolder(val_dir, transform=val_tfms)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )

    print(f"\nDataset summary:")
    print(f"Train size: {len(train_ds)}")
    print(f"Val size:   {len(val_ds)}")
    print(f"Num classes: {len(train_ds.classes)}")

    return train_loader, val_loader, train_ds.classes