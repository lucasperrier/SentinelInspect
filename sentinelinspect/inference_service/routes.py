"""HTTP routes: a thin translation layer over the Predictor.

Every route does three things -- validate the request, call the core, map
exceptions to status codes. No inference logic lives here, which is what keeps
the API and the CLI honest with each other.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status

from sentinelinspect import __version__
from sentinelinspect.inference.predictor import InvalidImageError, Predictor
from sentinelinspect.inference_service.dependencies import get_predictor, max_upload_bytes
from sentinelinspect.inference_service.logging import Timer, log_event
from sentinelinspect.inference_service.schemas import HealthResponse, Prediction

router = APIRouter()


@router.get("/health", response_model=HealthResponse, tags=["ops"])
def health(request_predictor: Predictor = Depends(get_predictor)) -> HealthResponse:
    """Readiness plus which weights are serving."""
    policy = request_predictor.review_policy
    return HealthResponse(
        status="ok",
        model_loaded=True,
        model_name=request_predictor.metadata.name,
        checkpoint_sha256=request_predictor.metadata.checkpoint_sha256,
        package_version=__version__,
        review_band=[policy.lower, policy.upper],
    )


@router.post(
    "/predict",
    response_model=Prediction,
    tags=["inference"],
    responses={
        400: {"description": "Empty or undecodable image"},
        413: {"description": "Upload exceeds the configured size limit"},
        415: {"description": "Content type is not an image"},
        503: {"description": "Model not loaded"},
    },
)
async def predict(
    file: UploadFile = File(..., description="Image to classify"),
    predictor: Predictor = Depends(get_predictor),
) -> Prediction:
    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Expected an image upload, got content type {file.content_type!r}.",
        )

    payload = await file.read()

    if not payload:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Uploaded file is empty."
        )

    limit = max_upload_bytes()
    if len(payload) > limit:
        raise HTTPException(
            status_code=413,  # CONTENT_TOO_LARGE; the starlette constant name differs across versions
            detail=f"Upload is {len(payload)} bytes; the limit is {limit} bytes.",
        )

    try:
        with Timer() as timer:
            prediction = predictor.predict_image(payload)
    except InvalidImageError as exc:
        # the caller sent something unreadable: their problem, not a server fault
        log_event("predict.invalid_image", filename=file.filename, error=str(exc))
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    log_event(
        "predict.ok",
        filename=file.filename,
        bytes=len(payload),
        label=prediction.predicted_label,
        confidence=round(prediction.confidence_score, 4),
        needs_review=prediction.needs_review,
        latency_ms=round(timer.elapsed_ms, 2),
    )
    return prediction
