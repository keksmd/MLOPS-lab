from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from app.core.config import settings
from app.core.local_model_settings import local_model_settings
from app.metrics.config import OpenRouterConfig
from app.metrics.llm.local_model_client import LocalModelLLMClient
from app.metrics.llm.openrouter_client import OpenRouterLLMClient
from app.metrics.local_model_config import LocalModelConfig

from ..config import PlanningConfig
from ..schemas import InferenceRequest, InferenceResult
from ..service import PlanningService

router = APIRouter(prefix="/planning", tags=["planning"])


def get_planning_service() -> PlanningService:
    """Build a planning service instance from centralized application settings."""
    provider = settings.PLANNING_LLM_PROVIDER

    if provider == "local":
        model_name = settings.LOCAL_MODEL_NAME
        auth_header = local_model_settings.basic_auth_header
        if not auth_header:
            raise HTTPException(
                status_code=500,
                detail="LOCAL_MODEL_USER/LOCAL_MODEL_PASSWORD are not configured",
            )
        llm_client = LocalModelLLMClient(
            LocalModelConfig(
                auth_header=auth_header,
                model_name=model_name,
                base_url=settings.LOCAL_MODEL_BASE_URL,
            )
        )
    else:
        api_key = settings.OPENROUTER_API_KEY
        if not api_key:
            raise HTTPException(
                status_code=500,
                detail="OPENROUTER_API_KEY is not configured",
            )
        model_name = settings.OPENROUTER_MODEL_NAME
        llm_client = OpenRouterLLMClient(
            OpenRouterConfig(
                api_key=api_key,
                model_name=model_name,
                base_url=settings.OPENROUTER_BASE_URL,
            )
        )

    planning_config = PlanningConfig(
        max_few_shot_examples=3,
        default_model_name=model_name,
        include_prompt_debug=False,
        include_raw_response=True,
        enforce_placeholder_rules=True,
    )
    return PlanningService(llm_client=llm_client, config=planning_config)


@router.post("/predict", response_model=InferenceResult)
def predict_planning(
    request: InferenceRequest,
    service: PlanningService = Depends(get_planning_service),
) -> InferenceResult:
    """Generate a structured planning prediction for a single task."""
    return service.predict(request)
