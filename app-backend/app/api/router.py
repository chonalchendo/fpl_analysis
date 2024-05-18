from api.handlers import value_prediction
from fastapi import APIRouter

router = APIRouter()

router.include_router(
    value_prediction.router,
    prefix="/value_prediction",
    tags=["value_prediction"],
)
