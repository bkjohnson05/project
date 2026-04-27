# Read Me

# Herbarium Image Classification

A deep learning system that classifies plant species from images using transfer learning with EfficientNet. The project explores how fine-tuning, data augmentation, and regularization impact model performance on a fine-grained classification task.

## What it Does

This project takes an input image of a plant and predicts its species using a convolutional neural network. The model outputs both top-1 and top-5 predictions, which is useful for fine-grained classification where multiple species may appear visually similar. The system is built using PyTorch and leverages a pretrained EfficientNet backbone that is adapted to the herbarium dataset through different training strategies.

## Quick Start

 1. Install dependencies:
    pip install -r requirements.txt

 2. Run an experiment:
    python -m experiments.run_baseline

(see SETUP.md for more)

## Video Links
 - Demo Video: 
 - Technical Walkthrough:

## Evaluation

### Results

| Model | Aug | WD | Top-1 Acc | Top-5 Acc | Loss |
|------|-----|----|-----------|-----------|------|
| Frozen No Aug | No | 0 | 0.72 | 0.936 | 1.23 |
| Frozen Aug | Yes | 0 | 0.603 | 0.894 | 1.61 |
| Frozen Aug + WD | Yes | 1e-4 | 0.63 | 0.91 | 1.47 |
| Partial FT No Aug | No | 0 | 0.776 | 0.966 | 0.75 |
| Partial FT Aug | Yes | 0 | 0.699 | 0.94 | 1.01 |
| Partial FT Aug + WD | Yes | 1e-4 | 0.705 | 0.951 | 1.01 |
| Full FT No Aug (Best) | No | 0 | **0.845** | **0.986** | **0.53** |
| Full FT Aug | Yes | 0 | 0.824 | 0.981 | 0.60 |
| Full FT Aug + WD | Yes | 1e-4 | 0.809 | 0.984 | 0.58 |

### Analysis

Fine-tuning the pretrained EfficientNet backbone significantly improved performance compared to using it as a frozen feature extractor. Moving from frozen to full fine-tuning increased top-1 accuracy from approximately 72% to 84.5%, demonstrating the importance of adapting learned feature representations.

Interestingly, data augmentation slightly reduced performance across experiments, suggesting that the applied transformations may have introduced noise or unrealistic variations for this dataset. Weight decay had only a minor impact, indicating that overfitting was not a major issue for the fully fine-tuned models.

Error analysis shows that the model struggles most with visually similar plant species, highlighting the challenge of fine-grained classification.

## Individual Contributions

This project was completed individually.