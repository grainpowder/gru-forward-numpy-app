from fastapi import APIRouter, Path, Query, HTTPException, status
from fastapi.requests import Request
from npgru.output import PredictionResponse, PredictionOutput
from npgru.predictor import PredictorTypes

ROUTER_PREFIX = "/predict"
router = APIRouter(prefix=ROUTER_PREFIX, tags=["predict"])


@router.get(
    path="/{predictor_type}",
    response_model=PredictionResponse
)
async def predict_category(
        request: Request,
        predictor_type: PredictorTypes = Path(...),
        title: str = Query(...),
        num_prediction: int = Query(..., alias="k")
):
    if predictor_type == PredictorTypes.NUMPY:
        result = request.app.state.predictor_numpy.predict(title, num_prediction)
    elif predictor_type == PredictorTypes.TENSORFLOW:
        result = request.app.state.predictor_tensorflow.predict(title, num_prediction)
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid predictor type : {predictor_type}"
        )
    outputs = [PredictionOutput(category=category, score=score) for category, score in result]
    return PredictionResponse(result=outputs)