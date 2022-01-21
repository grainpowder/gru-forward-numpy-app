from fastapi import APIRouter, Path, HTTPException, status
from fastapi.requests import Request
from npgru.predictor import PredictorTypes

ROUTER_PREFIX = "/health"
router = APIRouter(prefix=ROUTER_PREFIX, tags=["health check"])


@router.get("/ping")
async def ping() -> str:
    return "pong"


@router.get("/predictor/{predictor_type}")
async def check_predictor_loaded_on_startup(
        request: Request,
        predictor_type: PredictorTypes = Path(..., )
) -> str:
    if predictor_type == PredictorTypes.TENSORFLOW:
        predictor = request.app.state.predictor_tensorflow
    elif predictor_type == PredictorTypes.NUMPY:
        predictor = request.app.state.predictor_numpy
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid predictor type: {predictor_type}"
        )
    prediction = predictor.predict("Hello World", 3)
    return f"{prediction}"
