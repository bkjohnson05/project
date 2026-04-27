import torch
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from PIL import Image



def collect_predictions(model, loader, device):
    model.eval()
    results = []

    samples = loader.dataset.samples  # direct access

    ptr = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            preds = model(images).argmax(dim=1).cpu()

            batch_size = len(labels)

            for i in range(batch_size):
                path, _ = samples[ptr + i]

                results.append({
                    "path": path,
                    "true": int(labels[i].item()),
                    "pred": int(preds[i].item())
                })

            ptr += batch_size

    return results


def compute_confusion_matrix(results, num_classes):
    y_true = [r["true"] for r in results]
    y_pred = [r["pred"] for r in results]

    return confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))


def get_top_confusions(cm, top_k=5):
    confusions = []
    num_classes = cm.shape[0]

    for i in range(num_classes):
        for j in range(num_classes):
            if i != j and cm[i, j] > 0:
                confusions.append((i, j, int(cm[i, j])))

    confusions.sort(key=lambda x: x[2], reverse=True)
    return confusions[:top_k]


def get_top_mutual_confusions(cm, top_k=5):
    pairs = []
    num_classes = cm.shape[0]

    for i in range(num_classes):
        for j in range(i + 1, num_classes):
            score = cm[i, j] + cm[j, i]
            if score > 0:
                pairs.append((i, j, int(score)))

    pairs.sort(key=lambda x: x[2], reverse=True)
    return pairs[:top_k]


def plot_top_confusion_heatmap(cm, top_confusions, class_names, save_path=None):
    involved = sorted(set(
        [a for a, _, _ in top_confusions] +
        [b for _, b, _ in top_confusions]
    ))

    sub_cm = cm[np.ix_(involved, involved)]
    labels = [class_names[i] for i in involved]

    plt.figure(figsize=(min(12, 2*len(involved)), 6))
    sns.heatmap(sub_cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels)

    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Top Confused Classes")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)

    plt.show()

def group_examples(results):
    confused = defaultdict(list)
    correct = defaultdict(list)

    for r in results:
        if r["true"] == r["pred"]:
            correct[r["true"]].append(r["path"])
        else:
            confused[(r["true"], r["pred"])].append(r["path"])

    return confused, correct


def group_examples(results):
    confused = defaultdict(list)
    correct = defaultdict(list)

    for r in results:
        if r["true"] == r["pred"]:
            correct[r["true"]].append(r["path"])
        else:
            confused[(r["true"], r["pred"])].append(r["path"])

    return confused, correct


def show_confusion_with_reference(
    confused_dict,
    correct_dict,
    true_class,
    pred_class,
    class_names,
    n=4,
    save_path=None
):
    correct_paths = correct_dict.get(true_class, [])
    confused_paths = confused_dict.get((true_class, pred_class), [])

    if len(confused_paths) == 0 or len(correct_paths) == 0:
        print("Not enough examples")
        return

    n = min(n, len(correct_paths), len(confused_paths))

    plt.figure(figsize=(4 * n, 8))

    for i in range(n):
        img = Image.open(correct_paths[i]).convert("RGB")
        plt.subplot(2, n, i + 1)
        plt.imshow(img)
        plt.title(f"Correct: {class_names[true_class]}")
        plt.axis("off")

    for i in range(n):
        img = Image.open(confused_paths[i]).convert("RGB")
        plt.subplot(2, n, n + i + 1)
        plt.imshow(img)
        plt.title(f"True: {class_names[true_class]}\nPred: {class_names[pred_class]}")
        plt.axis("off")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)

    plt.show()


def analyze_top_confusions(
    model,
    val_loader,
    class_names,
    device,
    top_k=5,
    examples_per_pair=4,
    save_dir=None
):
    results = collect_predictions(model, val_loader, device)

    num_classes = len(class_names)
    cm = compute_confusion_matrix(results, num_classes)

    top_confusions = get_top_confusions(cm, top_k)
    confused_dict, correct_dict = group_examples(results)

    print("\nTop Confused Class Pairs:")
    for i, (a, b, count) in enumerate(top_confusions):
        print(f"{i+1}. {class_names[a]} → {class_names[b]} ({count})")

    plot_top_confusion_heatmap(
        cm,
        top_confusions,
        class_names,
        save_path=f"{save_dir}/confusion_heatmap.png" if save_dir else None
    )

    for a, b, count in top_confusions:
        print(f"\nAnalyzing: {class_names[a]} → {class_names[b]}")

        save_path = None
        if save_dir:
            save_path = f"{save_dir}/{class_names[a]}_{class_names[b]}.png"

        show_confusion_with_reference(
            confused_dict,
            correct_dict,
            a,
            b,
            class_names,
            n=examples_per_pair,
            save_path=save_path
        )

    return {
        "cm": cm,
        "top_confusions": top_confusions,
        "results": results,
        "confused_dict": confused_dict,
        "correct_dict": correct_dict
    }