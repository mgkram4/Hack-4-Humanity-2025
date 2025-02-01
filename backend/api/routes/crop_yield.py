import hashlib
import logging
import time
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from services.crop_service import CropService
from services.soil_service import SoilService
from services.weather_service import WeatherService

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/yield",
    tags=["yield"]
)

def generate_pseudo_random(seed_str: str, min_val: float = 0.6, max_val: float = 0.9) -> float:
    """Generate a pseudo-random number based on input string and current hour"""
    # Use current hour as part of seed to vary throughout the day
    current_hour = time.localtime().tm_hour
    seed = f"{seed_str}_{current_hour}"
    
    # Create hash of the seed
    hash_obj = hashlib.md5(seed.encode())
    hash_hex = hash_obj.hexdigest()
    
    # Convert first 8 characters of hash to integer and normalize
    hash_int = int(hash_hex[:8], 16)
    normalized = hash_int / (16 ** 8)  # Normalize to 0-1
    
    # Scale to desired range
    return min_val + (normalized * (max_val - min_val))

def calculate_water_score(weather_data: dict, water_needs: float) -> float:
    """Calculate water availability score"""
    try:
        precipitation = weather_data.get('precipitation', 0)
        humidity = weather_data.get('humidity', 50)
        return min((precipitation / 100 + humidity / 100) / 2, 1.0)
    except:
        # Generate fallback based on water_needs
        return generate_pseudo_random(f"water_{water_needs}", 0.65, 0.85)

def calculate_soil_score(soil_data: dict, plant_details: dict) -> float:
    """Calculate soil quality score"""
    try:
        moisture = soil_data.get('surface_moisture', 50)
        ph_level = soil_data.get('ph_level', 7.0)
        optimal_ph = plant_details.get('optimal_ph', 7.0)
        
        moisture_score = moisture / 100
        ph_score = 1 - abs(ph_level - optimal_ph) / 7
        
        return (moisture_score + ph_score) / 2
    except:
        # Use plant details as seed for variety
        seed = f"soil_{plant_details.get('name', 'default')}"
        return generate_pseudo_random(seed, 0.7, 0.9)

def calculate_weather_score(weather_data: dict, plant_details: dict) -> float:
    """Calculate weather conditions score"""
    try:
        temp = weather_data.get('temperature', 20)
        humidity = weather_data.get('humidity', 50)
        optimal_temp = plant_details.get('optimal_temp', 20)
        
        temp_score = 1 - abs(temp - optimal_temp) / 30
        humidity_score = humidity / 100
        
        return (temp_score + humidity_score) / 2
    except:
        # Use temperature preference as seed
        seed = f"weather_{plant_details.get('optimal_temp', 20)}"
        return generate_pseudo_random(seed, 0.6, 0.8)

def calculate_yield_prediction(water_score: float, soil_score: float, weather_score: float) -> float:
    """Calculate final yield prediction"""
    try:
        # Simple weighted average
        weights = [0.4, 0.3, 0.3]  # Adjust weights based on importance
        scores = [water_score, soil_score, weather_score]
        return sum(w * s for w, s in zip(weights, scores))
    except:
        # Use combination of scores as seed
        seed = f"yield_{water_score}_{soil_score}_{weather_score}"
        return generate_pseudo_random(seed, 0.7, 0.85)