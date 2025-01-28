import torch
import torch.nn as nn

class CNNClassifier(nn.Module):
    def __init__(self, input_shape=(3, 32, 32), num_classes=3):
        super(CNNClassifier, self).__init__()
        
        # Define CNN layers
        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        # Compute the flattened feature size dynamically
        with torch.no_grad():
            sample_input = torch.zeros(1, *input_shape)
            flat_size = self.conv_layers(sample_input).view(-1).shape[0]

        # Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Linear(flat_size, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # Flatten for FC layers
        x = self.fc_layers(x)
        return x

