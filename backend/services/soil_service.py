import os
from dotenv import load_dotenv

import requests

class SoilService:
    def __init__(self):
        # You should store this in environment variables
        load_dotenv()

        self.agro_api_key = os.getenv("AGRO_API_KEY", "your_api_key_here")
        self.agro_soil_url = "http://api.agromonitoring.com/agro/1.0/soil"
        self.agro_polygon_url = "http://api.agromonitoring.com/agro/1.0/polygons"

        self.geocoding_api_key = os.getenv("GEOCODING_API_KEY", "your_api_key_here")
        self.geocoding_base_url = "https://geocode.maps.co/search"

    # convert city into coordinates (latitude, longitude) using Geocoding API
    def cityToGeocode(self, city: str) -> (float, float):
        params = {
            "q": city,
            "api_key": self.geocoding_api_key
        }
        response = requests.get(self.geocoding_base_url, params=params)
        response.raise_for_status()
        response = response.json()
        lat = float(response[0]["lat"])
        lon = float(response[0]["lon"])
        return lat, lon
        
    # create polygon & get polygon_id based on the city provided
    def get_polygonId(self, city: str) -> str:
        lat, lon = self.cityToGeocode(city)
        # Define a bounding box around the city (area ~ 300 ha)
        square = [
            [lon - 0.01, lat + 0.01],  # Top-left
            [lon + 0.01, lat + 0.01],  # Top-right
            [lon + 0.01, lat - 0.01],  # Bottom-right
            [lon - 0.01, lat - 0.01],  # Bottom-left
            [lon - 0.01, lat + 0.01]   # Close the polygon
        ]
        print(square)
        headers = { "Content-Type": "application/json" }
        params = {
            "appid": self.agro_api_key,
        }
        
        # GeoJSON data
        geo_json = {  
            "name":f"{city}'s Polygon", 
             "geo_json": { 
                "type":"Feature",
                "geometry": {
                    "type":"Polygon", 
                    "coordinates":[square]      
                }   
            }
        }
        # create a polygon
        response = requests.post(self.agro_polygon_url, params=params, headers=headers, json=geo_json)
        response.raise_for_status()
        response = response.json()
        return response['id']
    
    def get_soil_data(self, city: str) -> dict:
        """
            Get soil data for a specific city
            Returns formatted data suitable for ML processing
        """
        polygon_id = self.get_polygonId(city)
        params = {
            "polyid": polygon_id,
            "appid": self.agro_api_key
        }

        # fetch data
        response = requests.get(self.agro_soil_url)
        response.raise_for_status()
        response = response.json()
        processed_data = {
            "moisture": response["moisture"],
            "surface_temperature": response["t0"],
            "temperature_on_10cm_depth": response["t10"],
            "timestamp": response["dt"],
        }

        # delete the polygon to save api calls
        requests.delete(f"{self.agro_polygon_url}/{polygon_id}", params=params.pop("polyid"))
        return processed_data

soil_service = SoilService()
print(soil_service.get_soil_data("London"))
