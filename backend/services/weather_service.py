import os
from typing import Any, Dict

import httpx
import asyncio
from fastapi import HTTPException
from dotenv import load_dotenv


class WeatherService:
    def __init__(self):
        # You should store this in environment variables
        load_dotenv()
        self.api_key = os.getenv("WEATHER_API_KEY", "your_api_key_here")
        self.base_url = "http://api.openweathermap.org/data/2.5/weather"

    async def get_weather_data(self, city: str) -> Dict[Any, Any]:
        """
        Get weather data for a specific city
        Returns formatted data suitable for ML processing
        """
        try:
            async with httpx.AsyncClient() as client:
                params = {
                    "q": city,
                    "appid": self.api_key,
                    "units": "metric"  # Use metric units
                }
                
                response = await client.get(self.base_url, params=params)
                response.raise_for_status()
                data = response.json()

                # Format data for ML processing
                processed_data = {
                    "temperature": data["main"]["temp"],
                    "humidity": data["main"]["humidity"],
                    "pressure": data["main"]["pressure"],
                    "wind_speed": data["wind"]["speed"],
                    "weather_condition": data["weather"][0]["main"],
                    "timestamp": data["dt"]
                }
                
                return processed_data

        except httpx.HTTPError as e:
            raise HTTPException(status_code=500, detail=f"Weather API error: {str(e)}")
        except KeyError as e:
            raise HTTPException(status_code=500, detail=f"Data processing error: {str(e)}")

# Usage example in FastAPI route:
# @app.get("/weather/{city}")
# async def get_weather(city: str):
#     weather_service = WeatherService()
#     return await weather_service.get_weather_data(city)
# print(asyncio.run(get_weather("London")))