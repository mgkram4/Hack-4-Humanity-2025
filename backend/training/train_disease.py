import torch
import torch.nn as nn
import torch.optim as optim
import joblib
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from backend.models.disease_detection.preprocessing import DDPreprocessor
from backend.models.disease_detection.model import CNNClassifier

class CNNTrainer:
    def __init__(self, dataframe, model, learning_rate=0.001):
        self.model = model
        self.data = self._prepare_data(dataframe)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

    def _prepare_data(self, dataframe):
        class ImageDataset(Dataset):
            def __init__(self, dataframe):
                self.dataframe = dataframe

            def __len__(self):
                return len(self.dataframe)

            def __getitem__(self, idx):
                image = self.dataframe.iloc[idx]['ImageData']
                label = self.dataframe.iloc[idx]['Label']
                image_tensor = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)
                label_tensor = torch.tensor(label, dtype=torch.long)
                return image_tensor, label_tensor

        return DataLoader(ImageDataset(dataframe), batch_size=16, shuffle=True)

    def train(self, epochs=10):
        """Train the CNN model."""
        self.model.train()
        for epoch in range(epochs):
            running_loss = 0.0
            for X_batch, y_batch in self.data:
                self.optimizer.zero_grad()
                outputs = self.model(X_batch)
                loss = self.criterion(outputs, y_batch)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(self.data):.4f}")

    def evaluate(self, dataloader):
        """Evaluate the CNN model."""
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for X_batch, y_batch in dataloader:
                outputs = self.model(X_batch)
                _, predicted = torch.max(outputs.data, 1)
                total += y_batch.size(0)
                correct += (predicted == y_batch).sum().item()
        print(f"Accuracy: {100 * correct / total:.2f}%")
    
    def save_model(self, path='cnn_model.joblib'):
        joblib.dump(self.model.state_dict(), path)
        print(f"Model saved to {path}")
    
    def load_model(self, path='cnn_model.joblib'):
        self.model.load_state_dict(joblib.load(path))
        self.model.eval()
        print(f"Model loaded from {path}")


# if __name__ == "__main__":
#     # Example standalone usage
#     input_shape = (3, 32, 32)
#     df = DDPreprocessor().df
#     model = CNNClassifier(input_shape=input_shape)
#     trainer = CNNTrainer(df, model)
#     trainer.train(epochs=5)