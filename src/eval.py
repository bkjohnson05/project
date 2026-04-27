import torch
import csv
from pathlib import Path
import json

def compute_topk_accuracy(outputs, labels, k=5):
    topk = outputs.topk(k, dim=1).indices
    correct = topk.eq(labels.view(-1, 1)).any(dim=1)
    return correct.float().sum().item()

def evaluate_model(model, loader, criterion, device):
    """
    Evaluate a model on a dataset.

    Returns:
        dict with loss, top1, top5, total samples
    """
    model.eval()

    total_loss = 0.0
    correct_top1 = 0
    correct_top5 = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)

            preds = outputs.argmax(dim=1)
            correct_top1 += (preds == labels).sum().item()
            correct_top5 += compute_topk_accuracy(outputs, labels, k=5)

            total += labels.size(0)

    return {
        "loss": total_loss / total if total > 0 else 0.0,
        "top1": correct_top1 / total if total > 0 else 0.0,
        "top5": correct_top5 / total if total > 0 else 0.0,
        "num_samples": total,
    }

def summarize_metrics(metrics, prefix=""):
    print(f"{prefix}Loss:  {metrics['loss']:.4f}")
    print(f"{prefix}Top-1: {metrics['top1']:.4f}")
    print(f"{prefix}Top-5: {metrics['top5']:.4f}")


def save_results(csv_path, run_name, metrics, config):
    """
    Save experiment results to CSV.
    """
    csv_path = Path(csv_path)
    write_header = not csv_path.exists()

    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)

        if write_header:
            writer.writerow([
                "run_name",
                "top1",
                "top5",
                "loss",
                "config"
            ])

        writer.writerow([
            run_name,
            metrics["top1"],
            metrics["top5"],
            metrics["loss"],
            json.dumps(config)
        ])


def evaluate_and_print(model, loader, criterion, device, name="Model"):
    metrics = evaluate_model(model, loader, criterion, device)

    print(f"\n{name}")
    summarize_metrics(metrics)

    return metrics

def compare_runs(run_a, run_b):
    print("\nComparison:")
    print(f"{'Metric':10s} | {'Run A':10s} | {'Run B':10s}")
    print("-" * 40)
    print(f"Top-1     | {run_a['top1']:.4f} | {run_b['top1']:.4f}")
    print(f"Top-5     | {run_a['top5']:.4f} | {run_b['top5']:.4f}")


def save_epoch_metrics(csv_path, run_name, epoch, metrics):
    csv_path = Path(csv_path)
    write_header = not csv_path.exists()

    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)

        if write_header:
            writer.writerow(["run_name", "epoch", "top1", "top5", "loss"])

        writer.writerow([
            run_name,
            epoch,
            metrics["top1"],
            metrics["top5"],
            metrics["loss"]
        ])

def evaluate_and_analyze(model, loader, criterion, device, analysis_fn):
    metrics = evaluate_model(model, loader, criterion, device)
    analysis = analysis_fn(model, loader, device)

    return metrics, analysis

def random_baseline(loader, num_classes, device):
    import torch
    correct = 0
    total = 0

    for _, labels in loader:
        labels = labels.to(device)
        preds = torch.randint(0, num_classes, labels.shape).to(device)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return correct / total