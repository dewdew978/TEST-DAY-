#import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler, Dataset
#from sklearn.metrics import accuracy_score
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.optim import Adam
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import torchvision.models as models

from torch.optim.lr_scheduler import CosineAnnealingLR
# --------------------------------------------------------------------------------------------
# 1. Transforms
# --------------------------------------------------------------------------------------------
import random
from PIL import Image
import cv2
import numpy as np
#ero/dialation
class RandomStrokeThickness:
    """Randomly dilate or erode PIL image strokes to simulate thin or bold characters."""
    def __init__(self, p=0.4):
        self.p = p

    def __call__(self, img):
        if random.random() > self.p:
            return img

        # Convert PIL to OpenCV Numpy Array
        img_np = np.array(img)
        
        # Pick kernel size (1x1 to 3x3 for fine adjustment)
        kernel_size = random.choice([2, 3])
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        
        # Randomly decide to thicken or thin out strokes
        operation = random.choice(['thicken', 'thin'])
        
        if operation == 'thicken':
            # For dark strokes on white background, erode expands the dark region
            img_np = cv2.erode(img_np, kernel, iterations=1)
        else:
            # Dilation shrinks dark regions, making strokes thinner
            img_np = cv2.dilate(img_np, kernel, iterations=1)

        return Image.fromarray(img_np)

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    #thin/bold char with ero/dialation
    RandomStrokeThickness(p=0.4),
    transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
    transforms.RandomRotation(degrees=(-25, 25)),
    transforms.ColorJitter(
        brightness=0.2, 
        contrast=0.2, 
        saturation=0.1
    ),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    #rand erasing
    transforms.RandomErasing(
        p=0.25, 
        scale=(0.02, 0.1), 
        ratio=(0.3, 3.3), 
        value=0
    ),
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# --------------------------------------------------------------------------------------------
# 2. Config & Reproducibility
# --------------------------------------------------------------------------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_CLASSES = 72
SEED = 42

torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

generator = torch.Generator().manual_seed(SEED)

# --------------------------------------------------------------------------------------------
# 3. Datasets & Split (Fixed: Separate transforms for train and test)
# --------------------------------------------------------------------------------------------
# Load datasets separately so test set doesn't get augmentations
class TransformedSubset(Dataset):
    """Wraps a Dataset subset to dynamically apply transformations."""
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.subset[index]
        x = x.convert('RGB')
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.subset)

# Load raw dataset once WITHOUT any transforms
raw_base_dataset = ImageFolder(root='dataset/round2')

train_size = int(0.8 * len(raw_base_dataset))
test_size = len(raw_base_dataset) - train_size

# Perform split on raw dataset
raw_train_subset, raw_test_subset = random_split(raw_base_dataset, [train_size, test_size], generator=generator)

# Wrap each subset with its dedicated transform
train_dataset = TransformedSubset(raw_train_subset, transform=train_transform)
test_dataset = TransformedSubset(raw_test_subset, transform=test_transform)

# --------------------------------------------------------------------------------------------
# Class Balancing (Uncomment if classes are imbalanced) Using weight
# --------------------------------------------------------------------------------------------
targets = [raw_base_dataset.samples[i][1] for i in raw_train_subset.indices]
class_counts = np.bincount(targets, minlength=NUM_CLASSES)
class_weights = 1.0 / (class_counts + 1e-6)
sample_weights = [class_weights[t] for t in targets]
sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

# --------------------------------------------------------------------------------------------
# 4. DataLoaders
# --------------------------------------------------------------------------------------------
# If using sampler above, set sampler=sampler and replace shuffle=True with shuffle=False
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False, sampler=sampler) 
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False,)

print(f"Total: {len(raw_base_dataset)} | Train: {len(train_dataset)} | Test: {len(test_dataset)}")

# --------------------------------------------------------------------------------------------
# 5. Model Architecture
# --------------------------------------------------------------------------------------------
weights = models.ResNet18_Weights.DEFAULT
model = models.resnet18(weights=weights)

# FIXED: Unfreeze weights so model can fine-tune properly on 72 classes
for param in model.parameters():
    param.requires_grad = True

#Add Drop-out 30% to the model
num_ftrs = model.fc.in_features
model.fc = nn.Sequential(  # type: ignore
    nn.Dropout(p=0.3),
    nn.Linear(num_ftrs, NUM_CLASSES)
)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
# Reduced learning rate to 1e-4 for transfer learning
EPOCHS = 15
optimizer = Adam(model.parameters(), lr=0.0003, weight_decay=1e-4)
scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)
# --------------------------------------------------------------------------------------------
# Early Stopping Helper Class
# --------------------------------------------------------------------------------------------
import copy
class EarlyStopping:
    def __init__(self, patience=4, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False
        self.best_model_wts = None

    def __call__(self, val_loss, model):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.best_model_wts = copy.deepcopy(model.state_dict())
        else:
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
# --------------------------------------------------------------------------------------------
# 6. Training & Validation Loop (Optimized for Speed)
# --------------------------------------------------------------------------------------------
early_stopping = EarlyStopping(patience=4, min_delta=0.001)

# Track loss and accuracy history for plotting
history = {
    'train_loss': [],
    'val_loss': [],
    'train_acc': [],
    'val_acc': []
}

print(f"Starting training on {device}...")

for epoch in range(EPOCHS):
    # Training Phase
    model.train()
    running_loss = torch.tensor(0.0, device=device)
    correct = torch.tensor(0, device=device)
    total = 0

    # mininterval=0.5 updates terminal smoothly with real-time stats
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1:02d}/{EPOCHS}", mininterval=0.5)
    #Batch
    for images, labels in pbar:
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        # Keep additions on GPU to prevent CPU-GPU synchronization bottlenecks
        batch_size = labels.size(0)
        running_loss += loss.detach() * batch_size
        _, predicted = outputs.max(1)
        total += batch_size
        correct += predicted.eq(labels).sum()

        # Real-time Loss and Accuracy display on tqdm progress bar
        current_avg_loss = (running_loss.item() / total)
        current_acc = (correct.item() / total) * 100
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'avg_loss': f'{current_avg_loss:.4f}',
            'acc': f'{current_acc:.2f}%'
        })
    #Out Batch
    scheduler.step()
    
    # Move totals to CPU only once per epoch for accuracy calculation
    train_acc = (correct.item() / total) * 100
    epoch_loss = (running_loss.item() / total)

    # Validation Phase
    model.eval()
    val_loss_running = torch.tensor(0.0, device=device)
    val_correct = torch.tensor(0, device=device)
    val_total = 0
    with torch.no_grad():
        for val_images, val_labels in test_loader:
            val_images, val_labels = val_images.to(device, non_blocking=True), val_labels.to(device, non_blocking=True)
            val_outputs = model(val_images)
            
            # Compute Validation Loss
            v_loss = criterion(val_outputs, val_labels)
            val_loss_running += v_loss.detach() * val_labels.size(0)
            
            _, val_pred = val_outputs.max(1)
            val_total += val_labels.size(0)
            val_correct += val_pred.eq(val_labels).sum()

    val_acc = (val_correct.item() / val_total) * 100
    val_loss = (val_loss_running.item() / val_total)

    # Save metrics to history
    history['train_loss'].append(epoch_loss)
    history['val_loss'].append(val_loss)
    history['train_acc'].append(train_acc)
    history['val_acc'].append(val_acc)

    print(f"Epoch {epoch+1:02d}/{EPOCHS} -> Train Loss: {epoch_loss:.4f} | Val Loss: {val_loss:.4f} | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%\n")

    early_stopping(val_loss, model)
    if early_stopping.early_stop:
        assert early_stopping.best_model_wts is not None
        print(f"\n[Early Stopping Triggered] Stopping early at epoch {epoch+1}.")
        # Restore best performing model weights before saving
        model.load_state_dict(early_stopping.best_model_wts)
        break
# --------------------------------------------------------------------------------------------
# 7. Evaluation & Model Saving
# --------------------------------------------------------------------------------------------
print("\nEvaluating on Test Set...")
model.eval()
test_correct = torch.tensor(0, device=device)
test_total = 0

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = outputs.max(1)
        test_total += labels.size(0)
        test_correct += predicted.eq(labels).sum()

test_acc = (test_correct.item() / test_total) * 100
print(f"Final Test Accuracy: {test_acc:.2f}%")

checkpoint = {
    'model_state': model.state_dict(),
    'class_to_idx': raw_base_dataset.class_to_idx
}

torch.save(checkpoint, 'model.pt')
print("Model saved successfully as model.pt!")

# --------------------------------------------------------------------------------------------
# 8. Plot & Save Training Curves (Loss & Accuracy)
# --------------------------------------------------------------------------------------------
if len(history['train_loss']) > 0:
    epochs_range = range(1, len(history['train_loss']) + 1)
    plt.figure(figsize=(14, 5))

    # Loss Curve
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, history['train_loss'], 'b-o', label='Train Loss')
    plt.plot(epochs_range, history['val_loss'], 'r-s', label='Val Loss')
    plt.title('Training & Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()

    # Accuracy Curve
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, history['train_acc'], 'b-o', label='Train Acc')
    plt.plot(epochs_range, history['val_acc'], 'g-s', label='Val Acc')
    plt.title('Training & Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()

    plt.tight_layout()
    plt.savefig('training_curves.png', dpi=300)
    print("Training curves saved successfully as 'training_curves.png'!")