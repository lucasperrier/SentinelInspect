"""HTTP-facing schemas.

`PredictResponse` is the contract from `inference.contracts`, not a copy of it.
Re-exporting rather than redefining is what guarantees the CLI and the API
return the same shape -- a parallel definition here would drift the first time
someone added a field to one of them.
"""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field

from sentinelinspect.inference.contracts import (
    BatchPrediction,
    ModelMetadata,
    Prediction,
    ReviewPolicy,
    ReviewReason,
)

__all__ = [
    "Prediction",
    "BatchPrediction",
    "ModelMetadata",
    "ReviewPolicy",
    "ReviewReason",
    "HealthResponse",
    "ErrorResponse",
]


class HealthResponse(BaseModel):
    """Liveness plus enough provenance to tell two deployments apart.

    A bare {"status": "ok"} cannot answer "which weights is production running?",
    which is the question you actually have during an incident.
    """

    model_config = ConfigDict(protected_namespaces=())

    status: str = Field(description="'ok' once the model is loaded and serving")
    model_loaded: bool
    model_name: Optional[str] = None
    checkpoint_sha256: Optional[str] = None
    package_version: str
    review_band: Optional[List[float]] = None


class ErrorResponse(BaseModel):
    detail: str
