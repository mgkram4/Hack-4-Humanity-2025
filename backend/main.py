from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.routes import resource, weather, soil
import uvicorn
from utils.middleware import configure_cors


app = FastAPI()
# add routes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(resource.router, prefix="/api/v1")
app.include_router(weather.router, prefix="/api/v1")
app.include_router(soil.router, prefix="/api/v1")

configure_cors(app)

@app.get("/test")
async def test():
    return {"message": "This is a test endpoint"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


    