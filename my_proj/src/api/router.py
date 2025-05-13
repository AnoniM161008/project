from dishka import FromComponent
from fastapi import APIRouter
from sqlalchemy.sql.annotation import Annotated
from starlette.exceptions import HTTPException
from starlette import status

from src.domain.nltk_use_cases.nltk import NLTKMethod, DTORequest

router = APIRouter(prefix="/nltk")

@router.get("", response_model=DTORequest)
async def get_sentiment(
    text: str,
    use_case: NLTKMethod = Annotated[NLTKMethod, FromComponent("nltk")],
) -> DTORequest:
    try:
        return use_case.analyze_sentiment_nltk(text=text)
    except Exception:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(Exception))

