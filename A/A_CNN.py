import torch
import torch.nn as nn
import torch.optim as optim
import os
from tqdm.auto import tqdm
from Utility.utils import TestModel, CM_Display

class A_CNN(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()

        # Feature Extractor Section
        self.block = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  
        )

        # Final Classifier Section
        self.classifier = nn.Sequential(
            nn.Flatten(),              # (64, 1, 1) → (64)
            nn.Linear(6272, 256),
            nn.ReLU(), 
            nn.Dropout(0.5), 
            nn.Linear(256, num_classes),  # 2 classes
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.block(x)
        x = self.classifier(x)
        return x
    
def A_CNN_Train(model, train_loader, val_loader, n_epochs, lr, device, criterion):
    train_losses = []
    val_losses = []
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(n_epochs):
        # Training
        model.train()
        running_loss, total = 0.0, 0

        # tqdm Progess Bar setup
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{n_epochs}", leave=False)
        for images, labels in loop:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            #forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            #backprop
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
            total += images.size(0)
        avg_train_loss = running_loss / total
        train_losses.append(avg_train_loss)

        # Validation
        model.eval()
        val_loss, val_total = 0.0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                #only forward pass since no training
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                val_total += images.size(0)
            avg_val_loss = val_loss / val_total
            val_losses.append(avg_val_loss)
        print(f"Epoch {epoch+1}/{n_epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
    return model, train_losses, val_losses
