from fastapi import FastAPI
from utils.middleware import configure_cors

app = FastAPI()

configure_cors(app)

@app.get("/test")
async def test():
    return {"message": "This is a test endpoint"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
    
    