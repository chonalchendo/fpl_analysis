from fastapi import APIRouter

from app.api.handlers import dropdowns, value_prediction

api_router = APIRouter()

api_router.include_router(
    value_prediction.router,
    prefix="/value_prediction",
    tags=["Value Prediction"],
)

api_router.include_router(
    dropdowns.router,
    prefix="/dropdowns",
    tags=["App dropdowns"],
)
