import numpy as np
from sklearn.preprocessing import StandardScaler

class ResourceDataPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_columns = [
            'temperature', 'humidity', 'soil_moisture',
            'soil_temperature', 'water_needs', 'nutrient_needs'
        ]
    
    def transform(self, data):
        features = self.extract_features(data)
        return self.scaler.transform(features)
    
    def extract_features(self, data):
        return np.array([[
            data['weather']['temperature'],
            data['weather']['humidity'],
            data['soil']['moisture'],
            data['soil']['surface_temperature'],
            data['plant']['water_needs'],
            data['plant']['soil_nutriments']
        ]])

