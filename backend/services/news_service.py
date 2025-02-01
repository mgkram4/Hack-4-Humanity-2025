import os
from typing import Dict

import fastapi
import httpx
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class NewsService:
    def __init__(self):
        self.news_base_url = "https://gnews.io/api/v4/search"
        self.news_api_key = os.getenv('GNEWS_API_KEY')
        
        if not self.news_api_key:
            raise ValueError("GNEWS_API_KEY not found in environment variables")
    
    async def get_news(self, country: str, category: str = None) -> Dict:
        """
        Fetch news using GNews API
        
        Args:
            country (str): Country code (e.g., 'us', 'gb')
            category (str, optional): News category
            
        Returns:
            Dict: News articles data
        """
        params = {
            "token": self.news_api_key,
            "lang": "en",  # English language news
            "country": country,
            "max": 10,  # Number of articles to return
            "q": "agriculture"  # Search term
        }
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(self.news_base_url, params=params)
                response.raise_for_status()
                data = response.json()
                
                print(f"API Response: {data}")  # Debug logging
                
                articles = data.get("articles", [])
                transformed_articles = [{
                    "title": article.get("title"),
                    "description": article.get("description"),
                    "url": article.get("url"),
                    "publishedAt": article.get("publishedAt"),
                    "source": {"name": article.get("source", {}).get("name")}
                } for article in articles]
                
                return {"articles": transformed_articles}
        except httpx.HTTPError as e:
            print(f"HTTP Error: {str(e)}")
            raise fastapi.HTTPException(status_code=500, detail=f"News API error: {str(e)}")
        except ValueError as e:
            print(f"Value Error: {str(e)}")
            raise fastapi.HTTPException(status_code=500, detail="Invalid response from News API")

        
