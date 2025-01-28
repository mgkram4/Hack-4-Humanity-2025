import asyncio
import torch.optim as optim
import torch.nn as nn
from models.resource_optimization.model import ResourceOptimizationModels
from models.resource_optimization.preprocessing import ResourceDataPreprocessor
from torch.utils.data import DataLoader
import joblib

async def train_models():
    # Initialize
    data_loader = DataLoader()
    preprocessor = ResourceDataPreprocessor()
    models = ResourceOptimizationModels()

# Collect and preprocess data
    raw_data = await data_loader.collect_training_data(100)
    X_train, X_test, y_train, y_test = preprocessor.prepare_training_data(raw_data)

# Train Random Forest (Water Model)
    models.water_model.fit(X_train, y_train['water'])
    print(f"Water Model Score: {models.water_model.score(X_test, y_test['water'])}")

# Train XGBoost (Nutrient Model)
    models.nutrient_model.fit(X_train, y_train['nutrient'])
    print(f"Nutrient Model Score: {models.nutrient_model.score(X_test, y_test['nutrient'])}")

# Train LSTM (Growth Model)
    sequence_data = preprocessor.prepare_lstm_data(X_train)
    optimizer = optim.Adam(models.growth_model.parameters())
    criterion = nn.MSELoss()

    for epoch in range(100):
        optimizer.zero_grad()
        outputs = models.growth_model(sequence_data)
        loss = criterion(outputs.squeeze(), nn.FloatTensor(y_train['growth']))
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f'Epoch [{epoch}/100], Loss: {loss.item():.4f}')

# Save models
    joblib.dump(models.water_model, 'models/water_model.joblib')
    joblib.dump(models.nutrient_model, 'models/nutrient_model.joblib')
    nn.save(models.growth_model.state_dict(), 'models/growth_model.pth')
    joblib.dump(preprocessor.scaler, 'models/scaler.joblib')

if __name__ == "main":
    asyncio.run(train_models())