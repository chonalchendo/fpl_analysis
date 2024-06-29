from fastapi import APIRouter

from app.schemas import Dropdowns
from app.services.database import get_dropdowns

router = APIRouter()


@router.get(
    "/get", description="Get values for dropdowns in the app", response_model=Dropdowns
)
async def dropdowns() -> Dropdowns:
    return await get_dropdowns()
