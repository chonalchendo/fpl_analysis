import uvicorn

# from app.api.router import router
from api.router import router
from fastapi import FastAPI

app = FastAPI(title="Football API", description="API for football data", version="1.0")

app.include_router(router, prefix="/api/v1")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
