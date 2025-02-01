import logging

from api.routes.crop_yield import router as crop_yield_router
from api.routes.disease import router as disease_router
# Import routers directly
from api.routes.resource import router as resource_router
from api.routes.soil import router as soil_router
from api.routes.weather import router as weather_router
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from models.yeild.yeild import CropRecommender
from utils.middleware import configure_cors

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Initialize CropRecommender with proper configuration
try:
    app.state.crop_recommender = CropRecommender()
    logger.info("CropRecommender initialized successfully")
except Exception as e:
    logger.error(f"Error initializing CropRecommender: {str(e)}")
    raise HTTPException(status_code=500, detail=f"Failed to initialize CropRecommender: {str(e)}")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add routes with renamed imports
app.include_router(resource_router, prefix="/api/v1")
app.include_router(weather_router, prefix="/api/v1")
app.include_router(soil_router, prefix="/api/v1")
app.include_router(crop_yield_router, prefix="/api/v1")
app.include_router(disease_router, prefix="/api/v1")

# Remove duplicate CORS configuration
# configure_cors(app)  # This is redundant since we already configured CORS above

@app.get("/health")
async def health_check():
    """
    Health check endpoint that includes model status
    """
    return {
        "status": "healthy",
        "models": {
            "crop_recommender": "trained" if app.state.crop_recommender.is_trained else "untrained"
        }
    }

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Error handling request: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc)
        }
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)