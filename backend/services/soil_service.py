import os

import requests
from dotenv import load_dotenv


class SoilService:
    def __init__(self):
        load_dotenv()
        self.geocoding_api_key = os.getenv("GEOCODING_API_KEY", "your_api_key_here")
        self.geocoding_base_url = "https://geocode.maps.co/search"
        self.soil_base_url = "https://power.larc.nasa.gov/api/temporal/daily/point"
        
    def cityToGeocode(self, city: str) -> tuple[float, float]:
        params = {"q": city, "api_key": self.geocoding_api_key}
        response = requests.get(self.geocoding_base_url, params=params)
        response.raise_for_status()
        data = response.json()
        
        if not data:
            raise Exception(f"No coordinates found for city: {city}")
            
        return float(data[0]["lat"]), float(data[0]["lon"])

    def get_soil_data(self, city: str) -> dict:
        try:
            lat, lon = self.cityToGeocode(city)
            
            params = {
                "parameters": "GWETPROF,GWETROOT,GWETTOP,TSOIL1,TSOIL2",
                "community": "AG",
                "longitude": lon,
                "latitude": lat,
                "start": "20240130",
                "end": "20240131",
                "format": "JSON"
            }
            
            response = requests.get(self.soil_base_url, params=params)
            response.raise_for_status()
            data = response.json()
            
            if "properties" not in data:
                raise Exception(f"No soil data available for coordinates: {lat}, {lon}")
                
            daily_data = data["properties"]["parameter"]
            current_date = list(daily_data["TSOIL1"].keys())[0]  # Get most recent date
            
            return {
                "surface_moisture": round(daily_data["GWETTOP"][current_date] * 100, 1),
                "root_moisture": round(daily_data["GWETROOT"][current_date] * 100, 1),
                "deep_moisture": round(daily_data["GWETPROF"][current_date] * 100, 1),
                "surface_temp": round(daily_data["TSOIL1"][current_date], 1),
                "subsurface_temp": round(daily_data["TSOIL2"][current_date], 1),
                "location": {"lat": lat, "lon": lon}
            }
            
        except Exception as e:
            raise Exception(f"Failed to get soil data: {str(e)}")