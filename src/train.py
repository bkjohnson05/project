import torch
from pathlib import Path
import copy
import time


CHECKPOINT_DIR = Path("checkpoints")
CHECKPOINT_DIR.mkdir(exist_ok=True)


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()

    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total

    return avg_loss, accuracy

def compute_topk_accuracy(outputs, labels, k=5):
    topk = outputs.topk(k, dim=1).indices
    correct = topk.eq(labels.view(-1, 1)).any(dim=1)
    return correct.float().sum().item()

def evaluate_epoch(model, loader, criterion, device):
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

    avg_loss = total_loss / total
    top1_acc = correct_top1 / total
    top5_acc = correct_top5 / total

    return avg_loss, top1_acc, top5_acc

def save_checkpoint(model, optimizer, epoch, val_loss, val_top1, run_name):
    path = CHECKPOINT_DIR / f"{run_name}_epoch{epoch}.pth"

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "val_loss": val_loss,
        "val_top1": val_top1,
        "run_name": run_name,
    }

    torch.save(checkpoint, path)
    print(f"Saved checkpoint: {path}")



def train_loop(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    device,
    num_epochs,
    run_name,
    save_best_only=True,
):
    start_time = time.time()
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_top1": [],
        "val_top5": [],
    }

    best_val_top1 = float("-inf")
    best_epoch = -1
    best_model_state = None

    for epoch in range(1, num_epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )

        val_loss, val_top1, val_top5 = evaluate_epoch(
            model, val_loader, criterion, device
        )

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_top1"].append(val_top1)
        history["val_top5"].append(val_top5)

        print(f"\n[{run_name}] Epoch {epoch}/{num_epochs}")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"  Val Loss:   {val_loss:.4f}")
        print(f"  Val Top-1:  {val_top1:.4f} | Val Top-5: {val_top5:.4f}")

        improved = val_top1 > best_val_top1

        if improved:
            best_val_top1 = val_top1
            best_epoch = epoch
            best_model_state = copy.deepcopy(model.state_dict())

            if save_best_only:
                save_checkpoint(model, optimizer, epoch, val_loss, val_top1, run_name)

        if not save_best_only:
            save_checkpoint(model, optimizer, epoch, val_loss, val_top1, run_name)

    # restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    results = {
        "run_name": run_name,
        "model": model,
        "epochs": num_epochs,
        "time_per_epoch": (time.time() - start_time) / num_epochs,
        "total_time": time.time() - start_time,
        "history": history,
        "best_val_top1": best_val_top1,
        "best_val_top5": max(history["val_top5"]) if history["val_top5"] else None,
        "best_epoch": best_epoch,
    }

    save_checkpoint(model, optimizer, num_epochs, val_loss, val_top1, f"{run_name}_final")

    end_time = time.time()
    print(f"Training time: {end_time - start_time:.2f} seconds")
    plot_curves(history)
    return results


def plot_curves(history):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history["train_loss"], label="train")
    axes[0].plot(history["val_loss"], label="val")
    axes[0].set_title("Loss")
    axes[0].legend()

    axes[1].plot(history["train_acc"], label="train")
    axes[1].plot(history["val_top1"], label="val top1")
    axes[1].plot(history["val_top5"], label="val top5")
    axes[1].set_title("Accuracy")
    axes[1].legend()

    plt.show()