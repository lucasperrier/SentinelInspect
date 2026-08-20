"""The prediction contract.

Everything that produces a prediction -- the CLI, the HTTP route, offline
evaluation -- returns one of these. Defining it here rather than in the service
layer is what stops the transport from inventing its own shape: adding a field
for the API would otherwise leave the CLI behind, and the two would drift.

These are Pydantic models rather than dataclasses so FastAPI can serialise them
directly and generate the OpenAPI schema from the same definition the CLI uses.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

# Index order is fixed by the label encoding in the datamodule: non_crack -> 0,
# crack -> 1. Changing this silently inverts every prediction.
CLASS_NAMES: tuple[str, str] = ("no_crack", "crack")
POSITIVE_INDEX = 1


class ReviewReason(StrEnum):
    """Why a prediction was routed to a human.

    A bare boolean would tell an operator that review is needed but not what
    triggered it, which matters when tuning the policy later.
    """

    LOW_CONFIDENCE = "low_confidence"


class ReviewPolicy(BaseModel):
    """The triage rule: when is the model not trusted to decide alone?

    Implemented as an abstain band around the decision boundary rather than a
    single floor. A one-sided rule would only catch uncertainty on one class;
    an image at p(crack)=0.52 and one at 0.48 are equally uncertain and both
    belong in front of a person.

    Defaults are tuned on the validation split, not guessed -- see
    `sentinelinspect.evaluation.metrics.tune_review_band`.
    """

    model_config = ConfigDict(extra="forbid")

    lower: float = Field(default=0.35, ge=0.0, le=1.0)
    upper: float = Field(default=0.65, ge=0.0, le=1.0)

    def model_post_init(self, __context) -> None:
        if self.lower > self.upper:
            raise ValueError(f"lower ({self.lower}) must not exceed upper ({self.upper})")

    def evaluate(self, positive_probability: float) -> tuple[bool, ReviewReason | None]:
        if self.lower <= positive_probability <= self.upper:
            return True, ReviewReason.LOW_CONFIDENCE
        return False, None

    @property
    def width(self) -> float:
        return self.upper - self.lower


class ModelMetadata(BaseModel):
    """Provenance for a prediction.

    Without this a stored prediction cannot be explained six months later: you
    know what was said but not which weights said it.
    """

    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    name: str = Field(description="Registry key, e.g. 'resnet50'")
    backbone: str = Field(description="timm backbone actually instantiated")
    checkpoint_path: str
    checkpoint_sha256: str = Field(description="First 16 hex characters of the checkpoint digest")
    package_version: str
    class_names: list[str]


class Prediction(BaseModel):
    """One decision about one image."""

    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    predicted_label: str
    predicted_index: int
    confidence_score: float = Field(ge=0.0, le=1.0, description="Probability of the predicted class")
    probabilities: dict[str, float]
    needs_review: bool
    review_reason: ReviewReason | None = None
    model_metadata: ModelMetadata
    latency_ms: float = Field(ge=0.0)


class BatchPrediction(BaseModel):
    """Several decisions plus the counts an operator actually wants."""

    model_config = ConfigDict(extra="forbid")

    predictions: list[Prediction]

    @property
    def review_count(self) -> int:
        return sum(1 for p in self.predictions if p.needs_review)

    @property
    def review_rate(self) -> float:
        return self.review_count / len(self.predictions) if self.predictions else 0.0
