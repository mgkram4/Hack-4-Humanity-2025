from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from models.resource_optimization.model import ResourceOptimizationModels
from models.resource_optimization.preprocessing import ResourceDataPreprocessor

router = APIRouter()
models = ResourceOptimizationModels()
preprocessor = ResourceDataPreprocessor()

class ResourceRequest(BaseModel):
    plant_data: dict
    weather_data: dict
    soil_data: dict

@router.post("/optimize")
async def optimize_resources(request: ResourceRequest):
    try:
        features = preprocessor.transform({
            'weather': request.weather_data,
            'soil': request.soil_data,
            'plant': request.plant_data
        })
        predictions = models.predict(features)
        return {"status": "success", "predictions": predictions}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
