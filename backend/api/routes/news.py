from typing import Optional

import fastapi
import uvicorn
from services.news_service import NewsService

app = fastapi.FastAPI()

@app.get("/news")
async def get_news(country: Optional[str] = "us", category: Optional[str] = None):
    """Get news for a specific country and category"""
    news_service = NewsService()
    return await news_service.get_news(country=country, category=category)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
