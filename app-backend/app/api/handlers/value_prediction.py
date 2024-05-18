from fastapi import APIRouter

router = APIRouter()


@router.get(
    "/",
    summary="Predict player valuation",
    description="Predict player market value based on player season statistics",
)
async def predict_valuation():
    return {"message": "Player prediction endpoint"}
