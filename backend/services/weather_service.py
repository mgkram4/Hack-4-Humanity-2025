import os
from datetime import datetime
from typing import Any, Dict

import httpx
from dotenv import load_dotenv
from fastapi import HTTPException


class WeatherService:
    def __init__(self):
        load_dotenv()
        self.api_key = os.getenv("WEATHER_API_KEY")
        self.base_url = "http://api.openweathermap.org/data/2.5"

    async def get_weather_data(self, city: str) -> Dict[Any, Any]:
        """
        Get current weather data for a specific city
        """
        try:
            async with httpx.AsyncClient() as client:
                params = {
                    "q": city,
                    "appid": self.api_key,
                    "units": "metric"
                }
                response = await client.get(f"{self.base_url}/weather", params=params)
                response.raise_for_status()
                data = response.json()
                
                # Create hourly data for the chart (using current data as we don't have hourly in free tier)
                current_hour = datetime.now().hour
                hourly_data = []
                for i in range(24):
                    hour = (current_hour + i) % 24
                    hourly_data.append({
                        "time": f"{hour:02d}:00",
                        "temperature": data["main"]["temp"],
                        "humidity": data["main"]["humidity"]
                    })

                # Format the response to match frontend expectations
                return {
                    "temperature": round(data["main"]["temp"]),
                    "humidity": data["main"]["humidity"],
                    "windSpeed": round(data["wind"]["speed"] * 3.6, 1),  # Convert m/s to km/h
                    "precipitation": 0,  # Not available in current weather
                    "hourlyData": hourly_data
                }
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    async def get_forecast_data(self, city: str) -> Dict[Any, Any]:
        """
        Get 7-day weather forecast for a specific city
        """
        try:
            async with httpx.AsyncClient() as client:
                params = {
                    "q": city,
                    "appid": self.api_key,
                    "units": "metric",
                    "cnt": 7
                }
                response = await client.get(f"{self.base_url}/forecast/daily", params=params)
                response.raise_for_status()
                data = response.json()

                # Format the forecast data for the frontend
                forecast = []
                for day in data["list"]:
                    date = datetime.fromtimestamp(day["dt"]).strftime("%a")
                    forecast.append({
                        "date": date,
                        "temperature": round(day["temp"]["day"])
                    })
                
                return forecast
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    async def get_weather_metrics(self) -> Dict[Any, Any]:
        """
        Get weather metrics for dashboard
        """
        try:
            # Example metrics - you might want to customize this
            metrics = {
                "current_temperature": 24,
                "humidity": 65,
                "wind_speed": 12,
                "precipitation": 0,
                "forecast": [
                    {"day": "Mon", "temp": 24},
                    {"day": "Tue", "temp": 23},
                    {"day": "Wed", "temp": 25},
                    {"day": "Thu", "temp": 22},
                    {"day": "Fri", "temp": 21},
                    {"day": "Sat", "temp": 23},
                    {"day": "Sun", "temp": 24}
                ]
            }
            return metrics
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))