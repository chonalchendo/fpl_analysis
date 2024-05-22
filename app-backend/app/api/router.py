from fastapi import APIRouter

from app.api.handlers import value_prediction

api_router = APIRouter()

api_router.include_router(
    value_prediction.router,
    prefix="/value_prediction",
    tags=["value_prediction"],
)
