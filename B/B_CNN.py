import torch
import torch.nn as nn

class B_CNN(nn.Module):
    def __init__(self, num_classes=8):
        super().__init__()

        # Block 1
        self.block = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2), 
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),   
        )

        # Final Classifier Section
        self.classifier = nn.Sequential(
            nn.Flatten(),              # (64, 1, 1) → (64)
            nn.Linear(3136, 256),
            nn.ReLU(), 
            nn.Dropout(0.5), 
            nn.Linear(256, num_classes),  # 8 classes
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.block(x)
        x = self.classifier(x)
        return x