import ast
import logging
import os
import re

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from model import CNNClassifier
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from torch.utils.data import DataLoader, Dataset

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_array_string(array_string):
    """Convert string representation of numpy array with ellipsis to actual array."""
    try:
        # Remove any whitespace at the start and end
        array_string = array_string.strip()
        
        # Split the string into lines
        lines = array_string.split('\n')
        
        # Remove empty lines
        lines = [line.strip() for line in lines if line.strip()]
        
        # Initialize lists to store the values
        all_values = []
        current_subarray = []
        
        for line in lines:
            # Skip lines with just brackets or ellipsis
            if line in ['[[[', ']]', ' [[', ' ]]', '...']:
                continue
                
            # Remove brackets and split by spaces
            line = line.replace('[', '').replace(']', '').strip()
            if line == '...':
                continue
                
            # Extract numbers using regex
            numbers = re.findall(r'[-+]?(?:\d*\.*\d+(?:[eE][-+]?\d+)?)', line)
            if numbers:
                # Convert strings to floats
                values = [float(x) for x in numbers]
                current_subarray.extend(values)
            
            # When we hit three values, we've completed an RGB pixel
            while len(current_subarray) >= 3:
                all_values.append(current_subarray[:3])
                current_subarray = current_subarray[3:]
        
        # Convert to numpy array and ensure correct shape (32, 32, 3)
        array = np.array(all_values, dtype=np.float32)
        if array.size != 32 * 32 * 3:
            missing_pixels = (32 * 32) - (array.shape[0])
            if missing_pixels > 0:
                # Pad with zeros if necessary
                padding = np.zeros((missing_pixels, 3), dtype=np.float32)
                array = np.vstack([array, padding])
        
        array = array.reshape(32, 32, 3)
        return array
        
    except Exception as e:
        logger.error(f"Error parsing array string: {str(e)}")
        raise

class PlantDiseaseDataset(Dataset):
    def __init__(self, data_frame):
        self.data = data_frame
        self.label_mapping = {
            "Powdery": 0,
            "Healthy": 1,
            "Rust": 2
        }

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        try:
            # Get the image data string and convert it to numpy array
            image_data_str = self.data.iloc[idx]['ImageData']
            
            # Add error handling for missing or invalid data
            if pd.isna(image_data_str) or not isinstance(image_data_str, str):
                raise ValueError(f"Invalid image data at index {idx}")
                
            image_array = parse_array_string(image_data_str)
            
            # Convert to PyTorch tensor and ensure correct shape
            image_tensor = torch.FloatTensor(image_array)
            image_tensor = image_tensor.permute(2, 0, 1)  # (H, W, C) -> (C, H, W)
            
            # Validate tensor shape
            expected_shape = (3, 32, 32)
            if image_tensor.shape != expected_shape:
                raise ValueError(f"Invalid tensor shape: got {image_tensor.shape}, expected {expected_shape}")
            
            # Get and convert label
            label = self.data.iloc[idx]['Label']
            if label not in self.label_mapping:
                raise ValueError(f"Invalid label at index {idx}: {label}")
                
            label = self.label_mapping[label]
            label = torch.tensor(label, dtype=torch.long)
            
            return image_tensor, label
            
        except Exception as e:
            logger.error(f"Error loading sample {idx}: {str(e)}")
            raise

def train_model():
    try:
        # Set device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")

        # Load dataset
        csv_path = os.path.join(os.path.dirname(__file__), 'plant_disease_dataset.csv')
        logger.info(f"Loading dataset from {csv_path}")
        
        df = pd.read_csv(csv_path)
        logger.info(f"Loaded {len(df)} samples")

        # Separate data by class for oversampling
        healthy_samples = df[df['Label'] == 'Healthy']
        powdery_samples = df[df['Label'] == 'Powdery']
        rust_samples = df[df['Label'] == 'Rust']

        # Oversample disease classes to balance dataset
        if len(powdery_samples) < len(healthy_samples):
            powdery_samples = resample(
                powdery_samples, 
                replace=True, 
                n_samples=len(healthy_samples),
                random_state=42
            )
        if len(rust_samples) < len(healthy_samples):
            rust_samples = resample(
                rust_samples, 
                replace=True, 
                n_samples=len(healthy_samples),
                random_state=42
            )

        # Combine all samples
        df = pd.concat([healthy_samples, powdery_samples, rust_samples])
        
        # Split dataset
        train_df, val_df = train_test_split(
            df, 
            test_size=0.2, 
            random_state=42, 
            stratify=df['Label']
        )
        logger.info(f"Training samples: {len(train_df)}, Validation samples: {len(val_df)}")

        # Create datasets
        train_dataset = PlantDiseaseDataset(train_df)
        val_dataset = PlantDiseaseDataset(val_df)

        # Calculate class weights for weighted loss
        class_counts = df['Label'].value_counts()
        total_samples = len(df)
        class_weights = {
            0: total_samples / (3 * class_counts['Powdery']),  # Increase weight for diseases
            1: total_samples / (6 * class_counts['Healthy']),  # Reduce weight for healthy
            2: total_samples / (3 * class_counts['Rust'])      # Increase weight for diseases
        }
        
        # Convert to tensor
        weight_tensor = torch.FloatTensor([
            class_weights[0],
            class_weights[1],
            class_weights[2]
        ]).to(device)

        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=32,
            shuffle=True,
            num_workers=0,
            pin_memory=torch.cuda.is_available()
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=32,
            shuffle=False,
            num_workers=0,
            pin_memory=torch.cuda.is_available()
        )

        # Initialize model
        model = CNNClassifier(input_shape=(3, 32, 32), num_classes=3)
        model = model.to(device)

        # Loss and optimizer with weighted loss
        criterion = nn.CrossEntropyLoss(weight=weight_tensor)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='max', 
            factor=0.1, 
            patience=3, 
            verbose=True
        )

        # Training loop
        num_epochs = 100
        best_val_acc = 0.0
        early_stopping_patience = 10
        epochs_without_improvement = 0

        for epoch in range(num_epochs):
            # Training phase
            model.train()
            train_loss = 0.0
            correct = 0
            total = 0

            for batch_idx, (inputs, targets) in enumerate(train_loader):
                try:
                    inputs, targets = inputs.to(device), targets.to(device)
                    
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    loss.backward()
                    optimizer.step()

                    train_loss += loss.item()
                    _, predicted = outputs.max(1)
                    total += targets.size(0)
                    correct += predicted.eq(targets).sum().item()

                    if batch_idx % 10 == 0:
                        logger.info(
                            f'Epoch: {epoch+1}/{num_epochs} | '
                            f'Batch: {batch_idx}/{len(train_loader)} | '
                            f'Loss: {loss.item():.4f}'
                        )
                except Exception as e:
                    logger.error(f"Error in training batch {batch_idx}: {str(e)}")
                    continue

            # Calculate and log training metrics
            train_acc = 100. * correct / total if total > 0 else 0
            avg_train_loss = train_loss / len(train_loader) if len(train_loader) > 0 else float('inf')
            
            # Validation phase
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0

            with torch.no_grad():
                for inputs, targets in val_loader:
                    try:
                        inputs, targets = inputs.to(device), targets.to(device)
                        outputs = model(inputs)
                        loss = criterion(outputs, targets)

                        val_loss += loss.item()
                        _, predicted = outputs.max(1)
                        val_total += targets.size(0)
                        val_correct += predicted.eq(targets).sum().item()
                    except Exception as e:
                        logger.error(f"Error in validation: {str(e)}")
                        continue

            # Calculate validation metrics
            val_acc = 100. * val_correct / val_total if val_total > 0 else 0
            avg_val_loss = val_loss / len(val_loader) if len(val_loader) > 0 else float('inf')

            logger.info(
                f'Epoch {epoch+1}/{num_epochs}:\n'
                f'Train Loss: {avg_train_loss:.4f} | '
                f'Train Acc: {train_acc:.2f}%\n'
                f'Val Loss: {avg_val_loss:.4f} | '
                f'Val Acc: {val_acc:.2f}%'
            )

            # Update learning rate
            scheduler.step(val_acc)

            # Save best model and check for early stopping
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                model_save_path = os.path.join(os.path.dirname(__file__), 'model.pth')
                torch.save(model.state_dict(), model_save_path)
                logger.info(f'Saved best model with validation accuracy: {val_acc:.2f}%')
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= early_stopping_patience:
                    logger.info(f'Early stopping triggered after {epoch + 1} epochs')
                    break

        logger.info(f'Training completed. Best validation accuracy: {best_val_acc:.2f}%')

    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        train_model()
    except Exception as e:
        logger.error(f"Training script failed: {str(e)}")
        raise