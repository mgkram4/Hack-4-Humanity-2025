import numpy as np
from fastapi import HTTPException
from sklearn.ensemble import RandomForestRegressor


class CropRecommender:
    def __init__(self):
        self.model = RandomForestRegressor()
        self.is_trained = False
        
        # Simple mock training data
        X = np.random.rand(100, 4)  # [temperature, humidity, rainfall, soil_quality]
        y = np.random.rand(100) * 100  # yield percentage
        self.model.fit(X, y)
        self.is_trained = True

    async def predict_for_location(self, city: str, crop_name: str = None):
        """Make predictions based on location and optionally a specific crop"""
        try:
            # Mock environmental data for demonstration
            mock_data = np.array([[25, 60, 100, 0.8]])  # Example values
            
            predicted_yield = self.model.predict(mock_data)[0] / 100.0
            
            return {
                "recommended_crop": crop_name or "wheat",
                "predicted_yield": predicted_yield,
                "water_score": 0.8,
                "soil_quality_score": 0.7,
                "weather_score": 0.75,
                "confidence_score": 0.85,
                "location": {
                    "city": city,
                    "coordinates": {"lat": 0, "lng": 0}  # Mock coordinates
                }
            }
            
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to generate prediction: {str(e)}"
            )
