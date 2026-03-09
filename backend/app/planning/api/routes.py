from __future__ import annotations

import os

from fastapi import APIRouter, Depends, HTTPException

from app.metrics.config import OpenRouterConfig
from app.metrics.llm.openrouter_client import OpenRouterLLMClient

from ..config import PlanningConfig
from ..schemas import InferenceRequest, InferenceResult
from ..service import PlanningService

router = APIRouter(prefix="/planning", tags=["planning"])


def get_planning_service() -> PlanningService:
    """
    Build a planning service instance from environment variables.

    Required environment variables:
    - OPENROUTER_API_KEY

    Optional environment variables:
    - OPENROUTER_MODEL_NAME
    """
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="OPENROUTER_API_KEY is not configured")

    model_name = os.getenv("OPENROUTER_MODEL_NAME", "meta-llama/llama-3.2-3b-instruct:free")
    llm_client = OpenRouterLLMClient(
        OpenRouterConfig(
            api_key=api_key,
            model_name=model_name,
        )
    )
    return PlanningService(llm_client=llm_client, config=PlanningConfig())


@router.post("/predict", response_model=InferenceResult)
def predict_planning(
    request: InferenceRequest,
    service: PlanningService = Depends(get_planning_service),
) -> InferenceResult:
    """Generate a structured planning prediction for a single task."""
    return service.predict(request)
