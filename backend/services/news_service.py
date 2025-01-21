import os
from typing import Dict

import fastapi
import httpx
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class NewsService:
    def __init__(self):
        self.news_base_url = "https://newsapi.org/v2/top-headlines"
        self.news_api_key = os.getenv('NEWS_API_KEY')
        
        if not self.news_api_key:
            raise ValueError("NEWS_API_KEY not found in environment variables")
    
    async def get_news(self, country: str, category: str = None) -> Dict:
        """
        Fetch news for a specific country and category.
        
        Args:
            country (str): Country code (e.g., 'us', 'gb')
            category (str, optional): News category
            
        Returns:
            Dict: News articles data
        """
        params = {
            "country": country,
            "apiKey": self.news_api_key
        }
        
        if category:
            params["category"] = category
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(self.news_base_url, params=params)
                response.raise_for_status()
                data = response.json()
                return {"articles": data.get("articles", [])}
        except httpx.HTTPError as e:
            raise fastapi.HTTPException(status_code=500, detail=f"News API error: {str(e)}")
        except ValueError as e:
            raise fastapi.HTTPException(status_code=500, detail="Invalid response from News API")

        
