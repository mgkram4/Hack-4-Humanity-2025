import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import joblib

class LSTMModel(nn.Module):
    def __init__(self, input_size=6, hidden_size=64):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])

class ResourceOptimizationModels:
    def __init__(self):
        self.water_model = RandomForestRegressor()
        self.nutrient_model = xgb.XGBRegressor()
        self.growth_model = LSTMModel()
        self.load_models()
    
    def load_models(self):
        try:
            self.water_model = joblib.load('models/water_model.joblib')
            self.nutrient_model = joblib.load('models/nutrient_model.joblib')
            self.growth_model.load_state_dict(torch.load('models/growth_model.pth'))
        except:
            print("Using untrained models")
    
    def predict(self, features):
        water_score = self.water_model.predict(features)
        nutrient_score = self.nutrient_model.predict(features)
        
        with torch.no_grad():
            features_tensor = torch.FloatTensor(features).unsqueeze(0)
            growth_pred = self.growth_model(features_tensor).item()
            
        return {
            'water_score': float(water_score),
            'nutrient_score': float(nutrient_score),
            'growth_prediction': float(growth_pred)
        }

