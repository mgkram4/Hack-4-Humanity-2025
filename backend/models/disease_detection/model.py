# backend/models/disease_detection/model.py
import torch
import torch.nn as nn


class CNNClassifier(nn.Module):
    def __init__(self, input_shape=(3, 32, 32), num_classes=3):
        super(CNNClassifier, self).__init__()
        
        # Define CNN layers
        self.conv_layers = nn.Sequential(
            # First conv block
            nn.Conv2d(input_shape[0], 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Second conv block
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Third conv block
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        # Calculate size of flattened features
        with torch.no_grad():
            sample_input = torch.zeros(1, *input_shape)
            flat_size = self.conv_layers(sample_input).view(-1).shape[0]
        
        # Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(flat_size, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        # Pass through conv layers
        x = self.conv_layers(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Pass through FC layers
        x = self.fc_layers(x)
        
        return x