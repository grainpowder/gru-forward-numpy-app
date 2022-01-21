from typing import List

from pydantic import BaseModel, Field


class PredictionOutput(BaseModel):
    category: int = Field(
        ...,
        title="Predicted Category",
        description="Index of predicted category",
        example=10
    )
    score: float = Field(
        ...,
        title="Prediction Score",
        description="0-1 standardized score representing likelihood that a title belongs to corresponding category",
        example=0.912345
    )


class PredictionResponse(BaseModel):
    result: List[PredictionOutput] = Field(
        ...,
        title="Prediction Result",
        description="List of multiple prediction outputs"
    )
