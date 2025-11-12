import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms, models
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
import numpy as np
from sklearn.metrics import classification_report

print("Starting script...")

train_path = r'C:\Users\lenovo\Downloads\Copy of InsPLAD-det\train'
val_path = r'C:\Users\lenovo\Downloads\Copy of InsPLAD-det\val'

if not os.path.exists(train_path):
    raise FileNotFoundError(f"Train path not found: {train_path}")
if not os.path.exists(val_path):
    raise FileNotFoundError(f"Val path not found: {val_path}")

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(0.5, 0.5, 0.5),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

print("Loading datasets...")
train_dataset = datasets.ImageFolder(train_path, transform=train_transform)
val_dataset = datasets.ImageFolder(val_path, transform=val_transform)

# Print class-to-index mapping for verification
print(f"Train set classes: {train_dataset.class_to_idx}")
print(f"Validation set classes: {val_dataset.class_to_idx}")

# Check your folder names to make sure class_to_idx matches your label names as expected
# For example, 'faulty' should map to the larger class index if that is the majority class

# It's important train and val have same class to index mapping
assert train_dataset.class_to_idx == val_dataset.class_to_idx, "Mismatch in train and val class indices!"

y_train = np.array([label for _, label in train_dataset.samples])
class_counts = np.bincount(y_train)
class_weights = 1. / class_counts
sample_weights = class_weights[y_train]
sampler = WeightedRandomSampler(weights=torch.DoubleTensor(sample_weights), num_samples=len(sample_weights), replacement=True)

batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", device)

model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
num_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(num_features, 2)
model = model.to(device)

criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32).to(device))
optimizer = optim.Adam(model.parameters(), lr=3e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=4)

max_epochs = 40
best_val_loss = float('inf')
patience, patience_counter = 10, 0

for epoch in range(max_epochs):
    start_time = time.time()
    print(f"\nEpoch {epoch+1}/{max_epochs} - Training")

    model.train()
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    print("Validating...")
    model.eval()
    val_loss, preds, truths = 0, [], []
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * images.size(0)
            preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
            truths.extend(labels.cpu().numpy())

    val_loss /= len(val_dataset)
    end_time = time.time()
    epoch_duration = end_time - start_time
    print(f'Epoch {epoch+1} completed in {epoch_duration:.2f} seconds')
    print(f'Validation Loss: {val_loss:.4f}')

    scheduler.step(val_loss)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), 'best_model.pth')
        patience_counter = 0
    else:
        patience_counter += 1
    if patience_counter >= patience:
        print("Early stopping triggered")
        break

print("\nTraining complete. Loading best model for evaluation...")
model.load_state_dict(torch.load('best_model.pth'))
print(classification_report(truths, preds, target_names=train_dataset.classes))










