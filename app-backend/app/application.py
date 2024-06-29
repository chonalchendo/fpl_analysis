from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.router import api_router
from app.core.config import get_settings


def get_app() -> FastAPI:
    app = FastAPI(
        title=get_settings().PROJECT_NAME,
        description=get_settings().DESCRIPTION,
        docs_url=f"{get_settings().API_V1_STR}/docs",
        redoc_url=f"{get_settings().API_V1_STR}/redoc",
        openapi_url=f"{get_settings().API_V1_STR}/openapi.json",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(api_router, prefix=get_settings().API_V1_STR)

    return app
