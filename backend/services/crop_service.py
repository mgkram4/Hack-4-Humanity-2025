import os

import requests
from dotenv import load_dotenv


class CropService:
    def __init__(self):
        # You should store this in environment variables
        load_dotenv()
        self.trefle_api_key = os.getenv("TREFLE_API_KEY", "your_api_key_here")
        self.base_url = "https://trefle.io/api/v1/plants"

    def get_plant_data(self, name: str) -> dict:
        """
            Get data for a particular plant
            Returns formatted data suitable for ML processing
        """
        params = {
            "token": self.trefle_api_key,
            "filter[common_name]": name.lower()
        }

        # make the first api call to get plant's general information
        response = requests.get(self.base_url, params=params)
        response.raise_for_status()
        response = response.json()
        if (not response["data"]):
            raise Exception("Sorry, the plant common name does not exist!")

        resource = response["data"][0]["links"]["plant"]

        self.base_url = f"https://trefle.io/{resource}"
        
        # make the second api call to get plant's specific information
        params.pop("filter[common_name]")
        response = requests.get(self.base_url, params=params)
        response.raise_for_status()
        response = response.json()

        processed_data = {
            "toxicity": response["data"]["main_species"]["specifications"].get("toxicity", 0),
            "light_needs": response["data"]["main_species"]["growth"].get("light", 0),
            "air_humidity_needs": response["data"]["main_species"]["growth"].get("atmospheric_humidity", 0),
            "soil_ph_range": [
                response["data"]["main_species"]["growth"].get("ph_minimum", 6.0),
                response["data"]["main_species"]["growth"].get("ph_maximum", 7.0)
            ],
            "temperature_range": [
                response["data"]["main_species"]["growth"].get("minimum_temperature", 15),
                response["data"]["main_species"]["growth"].get("maximum_temperature", 30)
            ],
            "growth_months": response["data"]["main_species"]["growth"].get("growth_months", []),
        }
        return processed_data

# Test code
crop_service = CropService()
print(crop_service.get_plant_data("Beach Strawberry"))
