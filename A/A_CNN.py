import torch
import torch.nn as nn


class A_CNN(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()

        # Feature Extractor Section
        self.block = nn.Sequential(
            nn.Conv2d(1, 5, kernel_size=3, padding=1),
            nn.BatchNorm2d(5), #5x28x28
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2), #5x14x14 
        )

        # Final Classifier Section
        self.classifier = nn.Sequential(
            nn.Flatten(),              
            nn.Linear(980, 256), 
            nn.ReLU(), 
            nn.Dropout(0.5), 
            nn.Linear(256, num_classes),  # 2 classes
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.block(x)
        x = self.classifier(x)
        return x
    
