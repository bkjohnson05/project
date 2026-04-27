import torch.nn as nn
from torchvision import models

def build_model(
    num_classes,
    mode="frozen",
    pretrained=True,
    dropout_p=0.2,
    device=None,
):
    weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
    model = models.efficientnet_b0(weights=weights)

    # Freeze or not
    if mode == "full_ft":
        for p in model.parameters():
            p.requires_grad = True
    else:
        for p in model.parameters():
            p.requires_grad = False

    # Replace classifier
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=dropout_p),
        nn.Linear(in_features, num_classes)
    )

    # Always train classifier
    for p in model.classifier.parameters():
        p.requires_grad = True

    # Partial fine-tuning
    if mode == "partial_ft":
        for p in model.features[-1].parameters():
            p.requires_grad = True

    elif mode not in ["frozen", "full_ft"]:
        raise ValueError(f"Unknown mode: {mode}")

    if device is not None:
        model = model.to(device)

    return model