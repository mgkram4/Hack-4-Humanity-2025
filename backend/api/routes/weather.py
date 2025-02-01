from datetime import datetime
from typing import Any, Dict

from fastapi import APIRouter, HTTPException
from services.weather_service import WeatherService

router = APIRouter(
    tags=["weather"]
)

weather_service = WeatherService()

@router.get("/weather/{city}")
async def get_weather(city: str) -> Dict[Any, Any]:
    """
    Get current weather data for a specific city
    """
    try:
        weather_data = await weather_service.get_weather_data(city)
        return {
            "status": "success",
            "data": weather_data
        }
    except Exception as e:
        if "city not found" in str(e).lower():
            raise HTTPException(status_code=404, detail="City not found")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/weather/forecast/{city}")
async def get_weather_forecast(city: str) -> Dict[Any, Any]:
    """
    Get 7-day weather forecast for a specific city
    """
    try:
        forecast_data = await weather_service.get_forecast_data(city)
        return {
            "status": "success",
            "data": forecast_data
        }
    except Exception as e:
        if "city not found" in str(e).lower():
            raise HTTPException(status_code=404, detail="City not found")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/weather/metrics")
async def get_weather_metrics() -> Dict[Any, Any]:
    """
    Get weather metrics for the dashboard
    """
    try:
        metrics = await weather_service.get_weather_metrics()
        return {
            "status": "success",
            "data": metrics
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))