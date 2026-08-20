"""The shared inference core.

Every prediction in this project -- CLI, offline evaluation, HTTP route -- goes
through `Predictor._probabilities`. That is the whole point of the class: the
model is loaded once, preprocessing comes from the same builder training uses,
and the triage rule is applied in one place. Three adapters over one core cannot
drift; three copies of the same twenty lines always do.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Iterable, List, Mapping, Optional, Sequence, Union

import numpy as np
import torch
from PIL import Image, UnidentifiedImageError

from sentinelinspect import __version__
from sentinelinspect.inference.contracts import (
    CLASS_NAMES,
    POSITIVE_INDEX,
    BatchPrediction,
    ModelMetadata,
    Prediction,
    ReviewPolicy,
)
from sentinelinspect.inference.model_loader import (
    checkpoint_fingerprint,
    load_model,
    resolve_device,
    resolve_model_config,
)
from sentinelinspect.preprocessing.transforms import build_inference_transforms

ImageInput = Union[str, Path, bytes, Image.Image, np.ndarray]


class InvalidImageError(ValueError):
    """The bytes given were not a readable image.

    A distinct type so the API can answer 400 rather than 500: a corrupt upload
    is the caller's problem, not a server fault.
    """


class Predictor:
    def __init__(
        self,
        model: torch.nn.Module,
        metadata: ModelMetadata,
        preprocessing: Optional[Mapping[str, Any]] = None,
        review_policy: Optional[ReviewPolicy] = None,
        device: Optional[torch.device] = None,
        class_names: Sequence[str] = CLASS_NAMES,
    ) -> None:
        self.model = model
        self.metadata = metadata
        self.review_policy = review_policy or ReviewPolicy()
        self.device = device or torch.device("cpu")
        self.class_names = list(class_names)
        # the SAME builder evaluation uses; never a locally assembled pipeline
        self.transform = build_inference_transforms(preprocessing)

    # ---- construction ----------------------------------------------------

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        model_config: Mapping[str, Any],
        preprocessing: Optional[Mapping[str, Any]] = None,
        review_policy: Optional[ReviewPolicy] = None,
        device: str | None = None,
    ) -> "Predictor":
        resolved_device = resolve_device(device)
        # what actually loaded, which may differ from what the config asked for
        resolved_config = resolve_model_config(checkpoint_path, model_config)
        model = load_model(checkpoint_path, model_config, device=resolved_device)
        metadata = ModelMetadata(
            name=str(resolved_config.get("name", "")),
            backbone=str(resolved_config.get("model", "")),
            checkpoint_path=str(checkpoint_path),
            checkpoint_sha256=checkpoint_fingerprint(checkpoint_path),
            package_version=__version__,
            class_names=list(CLASS_NAMES),
        )
        return cls(
            model=model,
            metadata=metadata,
            preprocessing=preprocessing,
            review_policy=review_policy,
            device=resolved_device,
        )

    @classmethod
    def from_runtime_config(cls, runtime) -> "Predictor":
        """Build from a validated RuntimeConfig, so the CLI, the service and
        evaluation all read the same configuration keys."""
        if not runtime.checkpoint_path:
            raise ValueError("checkpoint_path is required to build a Predictor")
        review = getattr(runtime, "review", None)
        return cls.from_checkpoint(
            checkpoint_path=runtime.checkpoint_path,
            model_config=runtime.model.model_dump(),
            preprocessing=runtime.preprocessing.model_dump() if runtime.preprocessing else None,
            review_policy=ReviewPolicy(**review.model_dump()) if review else None,
            device=runtime.device,
        )

    # ---- the one forward path -------------------------------------------

    @torch.inference_mode()
    def _probabilities(self, batch: torch.Tensor) -> torch.Tensor:
        """(B, C, H, W) preprocessed tensor -> (B, num_classes) probabilities."""
        if batch.ndim != 4:
            raise ValueError(f"Expected a 4D batch tensor, got shape {tuple(batch.shape)}")
        logits = self.model(batch.to(self.device))
        return torch.softmax(logits, dim=1).cpu()

    def _to_prediction(self, probabilities: torch.Tensor, latency_ms: float) -> Prediction:
        probs = probabilities.tolist()
        predicted_index = int(max(range(len(probs)), key=probs.__getitem__))
        needs_review, reason = self.review_policy.evaluate(probs[POSITIVE_INDEX])
        return Prediction(
            predicted_label=self.class_names[predicted_index],
            predicted_index=predicted_index,
            confidence_score=probs[predicted_index],
            probabilities={name: probs[i] for i, name in enumerate(self.class_names)},
            needs_review=needs_review,
            review_reason=reason,
            model_metadata=self.metadata,
            latency_ms=latency_ms,
        )

    # ---- adapters --------------------------------------------------------

    def predict_tensor(self, batch: torch.Tensor) -> List[Prediction]:
        """For data that is already preprocessed -- the evaluation dataloader.

        Sharing this with the image path is what guarantees offline metrics and
        served predictions come from identical arithmetic.
        """
        started = time.perf_counter()
        probabilities = self._probabilities(batch)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        per_sample = elapsed_ms / max(len(probabilities), 1)
        return [self._to_prediction(row, per_sample) for row in probabilities]

    def predict_image(self, image: ImageInput) -> Prediction:
        started = time.perf_counter()
        tensor = self._prepare(image).unsqueeze(0)
        probabilities = self._probabilities(tensor)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        return self._to_prediction(probabilities[0], elapsed_ms)

    def predict_images(self, images: Iterable[ImageInput]) -> BatchPrediction:
        images = list(images)
        if not images:
            return BatchPrediction(predictions=[])
        started = time.perf_counter()
        batch = torch.stack([self._prepare(image) for image in images])
        probabilities = self._probabilities(batch)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        per_sample = elapsed_ms / len(images)
        return BatchPrediction(
            predictions=[self._to_prediction(row, per_sample) for row in probabilities]
        )

    # ---- input handling --------------------------------------------------

    def _prepare(self, image: ImageInput) -> torch.Tensor:
        array = self._to_rgb_array(image)
        return self.transform(image=array)["image"]

    @staticmethod
    def _to_rgb_array(image: ImageInput) -> np.ndarray:
        """Accept the shapes callers actually have, and fail as InvalidImageError.

        A path from the CLI, raw bytes from an HTTP upload, a PIL image from a
        notebook, an array from a test. Normalising here keeps every caller from
        reimplementing it -- and reimplementing it slightly differently.
        """
        try:
            if isinstance(image, np.ndarray):
                pil = Image.fromarray(image)
            elif isinstance(image, Image.Image):
                pil = image
            elif isinstance(image, bytes):
                import io

                pil = Image.open(io.BytesIO(image))
            elif isinstance(image, (str, Path)):
                path = Path(image)
                if not path.exists():
                    raise InvalidImageError(f"Image not found: {path}")
                pil = Image.open(path)
            else:
                raise InvalidImageError(f"Unsupported image input type: {type(image).__name__}")
            return np.array(pil.convert("RGB"))
        except InvalidImageError:
            raise
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            raise InvalidImageError(f"Could not decode image: {exc}") from exc
