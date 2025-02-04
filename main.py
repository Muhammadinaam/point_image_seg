import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import VOCSegmentation
from torchvision.models.segmentation import FCN_ResNet50_Weights, fcn_resnet50
from torchvision.transforms import Compose, Normalize, Resize, ToTensor
from tqdm import tqdm


# Partial Cross Entropy Loss with Semi-Supervision
class PartialCrossEntropyLoss(nn.Module):
    def __init__(self):
        super(PartialCrossEntropyLoss, self).__init__()

    def forward(self, predictions, ground_truth, mask, full_mask):
        # Shape [batch, 21, H, W]
        predictions = torch.softmax(predictions, dim=1)

        # Ensure ground_truth and mask have correct shape
        if ground_truth.ndim == 4:
            ground_truth = ground_truth.squeeze(1)  # Shape [batch, H, W]
        if mask.ndim == 4:
            mask = mask.squeeze(1)  # Shape [batch, H, W]
        if full_mask.ndim == 4:
            full_mask = full_mask.squeeze(1)  # Shape [batch, H, W]

        # Apply focal loss only to masked regions
        focal_loss = -torch.log(predictions + 1e-7)  # Log probability
        focal_loss = focal_loss.gather(1, ground_truth.unsqueeze(
            1)).squeeze(1)  # Select correct class probs
        masked_loss = focal_loss * mask  # Only compute loss for labeled points
        loss = torch.sum(masked_loss) / (torch.sum(mask) + 1e-7)

        # Handle ignore index (255) in full mask
        valid_mask = full_mask != 255  # Create a mask for valid labels
        full_mask = full_mask.clone()  # Avoid modifying the original tensor
        # Set ignore indices to a safe class (e.g., 0)
        full_mask[~valid_mask] = 0

        # Semi-supervised learning: use full mask for additional supervision
        full_loss = nn.CrossEntropyLoss(ignore_index=255)(
            predictions, full_mask.long())

        return loss + 0.5 * full_loss  # Weighted combination


# Generate Sparse Point Labels
def generate_point_labels(masks, num_points=100):
    point_labels = torch.zeros_like(masks)
    for i in range(len(masks)):
        mask = masks[i]
        points = (mask > 0).nonzero(as_tuple=False)
        if len(points) > 0:
            sampled_points = points[torch.randperm(len(points))[:num_points]]
            point_labels[i, sampled_points[:, 0], sampled_points[:, 1]] = 1
    return point_labels


# Fix for VOCSegmentation Transform Issue
class VOCTransform:
    def __init__(self):
        self.transform = Compose([
            Resize((256, 256)),  # Resize images and masks to 256x256
            ToTensor(),
            Normalize(mean=[0.5], std=[0.5])
        ])

    def __call__(self, img, target):
        img = self.transform(img)
        target = Resize((256, 256))(target)  # Resize mask separately
        return img, torch.tensor(np.array(target), dtype=torch.long)


# Prepare DataLoader
def prepare_dataloader(batch_size=16, root="./data"):
    dataset_path = os.path.join(root, "VOCdevkit/VOC2012")

    if not os.path.exists(dataset_path):
        print("Downloading VOCSegmentation dataset...")
        dataset = VOCSegmentation(
            root=root, year='2012', image_set='train',
            download=True, transforms=VOCTransform())
    else:
        print("Dataset already exists. Skipping download.")
        dataset = VOCSegmentation(
            root=root, year='2012', image_set='train', download=False,
            transforms=VOCTransform())

    # subset data for 2 indices only for assessment as otherwise
    # it will take longer time
    dataset = torch.utils.data.Subset(dataset, indices=list(range(2)))
    training_fraction = 0.8

    train_size = int(training_fraction * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size])
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


# Simple Segmentation Model
def get_model(pretrained=True, learning_rate=0.001):
    weights = FCN_ResNet50_Weights.DEFAULT if pretrained else None
    # PASCAL VOC has 21 classes
    model = fcn_resnet50(weights=weights, num_classes=21)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    return model, optimizer


# Model Ensemble Function
def ensemble_predict(models, images):
    outputs = [model(images)['out'] for model in models]

    # Averaging predictions from multiple models
    avg_output = torch.mean(torch.stack(outputs), dim=0)
    return avg_output


# Model Evaluation Function
def evaluate_model(models, val_loader, device):
    for model in models:
        model.eval()
    total_loss = 0
    criterion = nn.CrossEntropyLoss(ignore_index=255)
    with torch.no_grad():
        for images, masks in tqdm(val_loader, desc="Evaluating Model"):
            images, masks = images.to(device), masks.to(device)

            # Use ensemble prediction
            outputs = ensemble_predict(models, images)
            loss = criterion(outputs, masks.long())
            total_loss += loss.item()
    return total_loss / len(val_loader)


# Training Function
def train_model(models, optimizers, train_loader, val_loader, epochs=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = PartialCrossEntropyLoss()
    for model in models:
        model.to(device)

    for epoch in range(epochs):
        for model in models:
            model.train()
        train_loss = 0
        start_time = time.time()

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch in progress_bar:
            # PASCAL VOC returns (image, mask) pairs
            images, full_masks = batch
            images = images.to(device)
            full_masks = full_masks.to(device)
            point_masks = generate_point_labels(
                full_masks)  # Generate sparse labels

            for model, optimizer in zip(models, optimizers):
                optimizer.zero_grad()
                outputs = model(images)['out']
                loss = criterion(outputs, point_masks, point_masks, full_masks)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            progress_bar.set_postfix(loss=train_loss / len(train_loader))

        epoch_time = time.time() - start_time
        print(f"Epoch [{epoch+1}/{epochs}], "
              f"Train Loss: {train_loss / len(train_loader):.4f}, "
              f"Time: {epoch_time:.2f}s")
    print("Training Complete!")


# Experiment and Report Generation
def run_experiments():
    experiments = [
        {"name": "Pretrained vs. Non-Pretrained", "pretrained": [True, False]},
        {"name": "Learning Rate Effect", "learning_rate": [
            0.01, 0.001, 0.0001]},
        {"name": "Epoch Count", "epochs": [5, 10, 20]}
    ]

    train_loader, val_loader = prepare_dataloader()
    report_lines = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for experiment in experiments:
        report_lines.append(f"Experiment: {experiment['name']}\n")

        for key, values in experiment.items():
            if key == "name":
                continue

            for value in values:
                report_lines.append(f"  - {key}: {value}\n")
                models, optimizers = zip(
                    *[get_model(
                        pretrained=True if key == "pretrained"
                        and value else False,
                        learning_rate=value if key == "learning_rate"
                        else 0.001)
                      for _ in range(3)])

                start_time = time.time()
                train_model(models, optimizers, train_loader,
                            val_loader,
                            epochs=value if key == "epochs" else 10)
                end_time = time.time()
                elapsed_time = end_time - start_time

                # Evaluate model performance
                # Use the first model for evaluation
                avg_loss = evaluate_model(models, val_loader, device)
                report_lines.append(
                    "    * Training Completed. Time Taken: "
                    f"{elapsed_time:.2f}s, Validation Loss: {avg_loss:.4f}\n")

    with open("experiment_report.txt", "w") as report:
        report.write("\n".join(report_lines))

    print("All experiments completed and report generated!")


# Main Execution
if __name__ == "__main__":
    run_experiments()
