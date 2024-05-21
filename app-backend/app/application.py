from api.router import router
from fastapi import FastAPI

from app.core.config import settings


def get_app() -> FastAPI:
    app = FastAPI(
        title=settings().PROJECT_NAME,
        description=settings().DESCRIPTION,
        docs_url=f"{settings().API_V1_STR}/docs",
        redoc_url=f"{settings().API_V1_STR}/redoc",
        openapi_url=f"{settings().API_V1_STR}/openapi.json",
    )
    app.include_router(router, prefix=settings().API_V1_STR)
    return app
