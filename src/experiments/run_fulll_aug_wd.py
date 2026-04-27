import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from data import get_dataloaders
from model import build_model
from train import train_loop
from eval import evaluate_model, save_results, random_baseline
from analysis import analyze_top_confusions

import torch.nn as nn
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RUN_NAME = "full_aug_weight_decay"


def main():
    train_loader, val_loader, class_names = get_dataloaders(
        split_root="data/herbarium_split",
        batch_size=64,
        image_size=224,
        num_workers=4,
        use_augmentation=True
    )

    model = build_model(
        num_classes=len(class_names),
        mode="full_ft",
        device=DEVICE
    )

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-3,
        weight_decay=1e-4
    )

    results = train_loop(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        DEVICE,
        num_epochs=5,
        run_name=RUN_NAME
    )


    baseline_acc = random_baseline(val_loader, len(class_names), DEVICE)
    print("Baseline accuracy:", baseline_acc)

    metrics = evaluate_model(model, val_loader, criterion, DEVICE)

    save_results("docs/results.csv", RUN_NAME, metrics, {
        "augmentation": True,
        "weight_decay": 1e-4
    })

    analyze_top_confusions(model, val_loader, class_names, DEVICE)


if __name__ == "__main__":
    main()