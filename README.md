# SentinelInspect

A visual inspection system for concrete surface imagery. It classifies an image as
`crack` or `no_crack`, reports a confidence score, and refers low-confidence cases to
manual review rather than returning an unreliable decision.

[![CI](https://github.com/lucasperrier/SentinelInspect/actions/workflows/ci.yaml/badge.svg)](https://github.com/lucasperrier/SentinelInspect/actions/workflows/ci.yaml)
![Python](https://img.shields.io/badge/python-3.11-blue)
![License](https://img.shields.io/badge/license-MIT-green)

---

## Contents

1. [Overview](#overview)
2. [Results](#results)
3. [System design](#system-design)
4. [Installation](#installation)
5. [Usage](#usage)
6. [HTTP API](#http-api)
7. [Container deployment](#container-deployment)
8. [Testing](#testing)
9. [Design decisions](#design-decisions)
10. [Limitations](#limitations)
11. [Repository structure](#repository-structure)
12. [Further documentation](#further-documentation)

---

## Overview

Automated visual inspection is rarely a pure classification problem. An inspection
workflow needs to know not only what the model predicts but how much that prediction can
be relied upon, and it needs a defined route for cases the model cannot settle.

SentinelInspect addresses this with three components:

- **A versioned dataset contract.** Training and evaluation read committed manifest and
  split files rather than scanning a directory, so the exact data behind any result is
  recorded and reproducible.
- **A single inference core.** The command-line tool, offline evaluation and the HTTP
  service all issue predictions through the same code path, which prevents divergence
  between measured and served behaviour.
- **An explicit triage policy.** Predictions whose confidence falls inside a calibrated
  band are flagged for human review instead of being decided automatically.

The project is intended to demonstrate the engineering practices that surround a model in
production — reproducible data handling, stable interfaces, tested failure modes,
packaging and deployment — rather than to advance classification accuracy on this task.

---

## Results

Model: ResNet-50 with a frozen backbone (linear probe), trained for three epochs on CPU
over 28,054 images. Evaluated once on a held-out test split of 5,889 images containing no
image present in training.

### Classification performance

| Metric | Test | Validation |
| --- | --- | --- |
| Accuracy | 0.9885 | 0.9846 |
| F1 | 0.9886 | 0.9844 |
| ROC AUC | 0.9990 | 0.9987 |
| Recall (crack) | 0.9899 | 0.9855 |
| Cross-entropy loss | 0.0707 | 0.0741 |

Test confusion matrix:

| | Predicted `no_crack` | Predicted `crack` |
| --- | --- | --- |
| **Actual `no_crack`** | 2872 | 38 |
| **Actual `crack`** | 30 | 2949 |

### Effect of the review policy

The confidence band was selected on the validation split and applied unchanged to test.

| Quantity | Value |
| --- | --- |
| Images referred to review | 383 of 5,889 (6.5%) |
| Model errors intercepted | 54 of 68 (79.4%) |
| Accuracy on automatically decided cases | 0.9975 (5,506 images) |

The band was selected to intercept 80% of errors on validation and intercepted 79.4% on
test, indicating that it generalised rather than fitting the selection split.

### Evaluation protocol

Splits are assigned by hashing the SHA256 of each image, so identical images cannot be
distributed across different splits. Validation of the dataset artifacts confirms zero
content-level overlap between train, validation and test.

An earlier revision of this repository reported 99.78% test accuracy on a split assigned
by filename. That split placed 8,482 images in both the training and test sets, because
the data directory contained a byte-identical duplicate of itself under a second name.
The figures above supersede it. The correction is documented in
[`docs/roadmap.md`](docs/roadmap.md).

---

## System design

```
data/raw/**  ──build_manifest──▶  manifest.csv  ──splitters──▶  train/val/test.csv
                                       │                              │
                                       └────── validate_dataset ──────┘
                                                     │
                                              CrackDataModule
                                                     │
                        ┌────────────────────────────┼───────────────────────┐
                        ▼                            ▼                       ▼
                    train.py                    Predictor              (same core)
              Hydra config + MLflow      load once · preprocess · forward
                        │                softmax · triage · typed contract
                        ▼                            │
                   checkpoint ─────────────────▶─────┴──▶ CLI · evaluate.py · FastAPI
```

Three principles determine the structure.

**The dataset is defined by files, not by a directory.** Training reads `train.csv`, a
committed artifact that can be diffed, hashed and shared. Adding images to `data/raw/`
has no effect until the manifest is deliberately rebuilt. The manifest records a path
relative to a configured root together with a content hash, and contains no
machine-specific absolute paths.

**Split assignment is a pure function of image content.** `assign_split(sha256)` uses no
random number generator, so an image receives the same split on any machine and adding
new images does not reassign existing ones. Class balance is therefore approximate rather
than exact; the measured deviation is below one percentage point.

**Prediction has one implementation.** `Predictor._probabilities` is the only forward pass
in the system. The CLI, offline evaluation and the HTTP route are adapters over it, which
is what allows a reported metric and a served response to be treated as equivalent.

Further detail is in [`docs/architecture.md`](docs/architecture.md).

---

## Installation

Requires Python 3.11 or later.

```bash
git clone https://github.com/lucasperrier/SentinelInspect.git
cd SentinelInspect
python -m venv .venv && source .venv/bin/activate
```

On Linux, the default PyTorch wheel includes the CUDA runtime, which adds several
gigabytes that are unused on a CPU-only machine or in a container. Install the CPU build
first if that applies:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

Then install the package. Optional dependency groups are available for the HTTP service
(`api`), training (`train`, which adds MLflow), explainability (`explain`, which adds
SHAP), and development (`dev`).

```bash
pip install -e ".[dev]"
```

The test suite constructs its own fixtures and requires neither the dataset nor a trained
model, so installation can be verified immediately:

```bash
pytest
```

---

## Usage

Six console scripts are installed with the package and may be run from any directory.

### Data preparation

```bash
# Build a manifest of data/raw/: relative path, label, dimensions and SHA256 per image
sentinelinspect-manifest

# Assign deterministic splits, keyed on image content
sentinelinspect-split

# Verify dataset integrity: readable files, no duplicates across splits, no label conflicts
sentinelinspect-validate \
  --manifest data/processed/manifests/manifest.csv \
  --train data/processed/splits/train.csv \
  --val   data/processed/splits/val.csv \
  --test  data/processed/splits/test.csv \
  --raw-root data/raw
```

### Training

```bash
pip install -e ".[train]"
sentinelinspect-train model.freeze_backbone=true trainer.max_epochs=3
```

Configuration is composed by Hydra from the groups in `configs/`. Model selection is a
configuration change rather than a code change:

```bash
sentinelinspect-train model=vit
```

### Evaluation

```bash
sentinelinspect-evaluate "checkpoint_path='runs/<run>/<file>.ckpt'" split=test
```

This writes an evaluation bundle containing `metrics.json`, a classification report, the
confusion matrix, and per-sample predictions with confidences and review flags. Retaining
per-sample outputs allows new metrics to be computed later without re-running the model.

### Single-image prediction

```bash
sentinelinspect-predict \
  "checkpoint_path='runs/<run>/<file>.ckpt'" \
  "image_path='data/raw/ccic/Positive/00001.jpg'"
```

> Hydra overrides containing `=` must be quoted, as checkpoint filenames include
> `epoch=NN`.

---

## HTTP API

```bash
pip install -e ".[api]"
SENTINELINSPECT_CHECKPOINT=runs/<run>/<file>.ckpt \
  uvicorn sentinelinspect.inference_service.app:app --port 8000
```

The model is loaded during application startup, before the first request is accepted. A
missing or unreadable checkpoint terminates the process rather than allowing the service
to report itself healthy while unable to serve. Interactive documentation is available at
`/docs`.

### `GET /health`

```json
{
  "status": "ok",
  "model_loaded": true,
  "model_name": "resnet50",
  "checkpoint_sha256": "ce03c54022ab5f65",
  "package_version": "0.1.0",
  "review_band": [0.23, 0.77]
}
```

The checkpoint fingerprint identifies which weights are in service, which a filename
cannot do reliably.

### `POST /predict`

```bash
curl -F "file=@image.jpg;type=image/jpeg" http://localhost:8000/predict
```

```json
{
  "predicted_label": "crack",
  "predicted_index": 1,
  "confidence_score": 0.9992,
  "probabilities": { "no_crack": 0.0008, "crack": 0.9992 },
  "needs_review": false,
  "review_reason": null,
  "model_metadata": {
    "name": "resnet50",
    "backbone": "resnet50",
    "checkpoint_path": "runs/.../model.ckpt",
    "checkpoint_sha256": "ce03c54022ab5f65",
    "package_version": "0.1.0",
    "class_names": ["no_crack", "crack"]
  },
  "latency_ms": 41.2
}
```

This is the same object returned by the command-line tool and produced by offline
evaluation; it is defined once, in `sentinelinspect/inference/contracts.py`.

| Status | Condition |
| --- | --- |
| 200 | Prediction returned |
| 400 | Empty upload, or content that cannot be decoded as an image |
| 413 | Upload exceeds `SENTINELINSPECT_MAX_UPLOAD_MB` (default 10) |
| 415 | Content type is not `image/*` |
| 422 | No file field present in the request |
| 503 | Model not loaded |

### Review policy

```yaml
# configs/review/default.yaml
lower: 0.23
upper: 0.77
```

A prediction is flagged when the probability of `crack` falls within the band. The band is
two-sided because a probability of 0.48 and one of 0.52 represent equal uncertainty; a
single confidence floor would flag only one of them. Bounds are selected on the validation
split subject to a maximum review rate, since widening the band always intercepts more
errors and an unconstrained search converges to the full interval.

---

## Container deployment

```bash
docker build -f docker/Dockerfile.api -t sentinelinspect-api .

docker run --rm -p 8000:8000 \
  -v "$(pwd)/runs:/models:ro" \
  -e SENTINELINSPECT_CHECKPOINT=/models/<run>/<file>.ckpt \
  sentinelinspect-api
```

The image is built in two stages so that build tooling is excluded from the runtime layer,
installs the CPU build of PyTorch, runs as an unprivileged user, and declares a
`HEALTHCHECK` against `/health`. Model weights are mounted at run time rather than
included in the image, so a new checkpoint does not require a rebuild.

---

## Testing

```bash
pytest
```

99 tests, unit and integration, completing in approximately seven seconds. Coverage is
directed at failure modes rather than at line count.

| Area | Behaviour under test |
| --- | --- |
| Splitting | Determinism; independence from row order; stability when the dataset grows; byte-identical images never separated |
| Validation | Content-level overlap invisible to a path comparison; severity separation; label conflicts |
| Contract | Review-band boundaries; rejection of inverted bands; label and index consistency |
| Inference | Equivalence across path, bytes, PIL and array inputs; equivalence of the image and tensor paths; typed errors for undecodable input; single model load |
| Metrics | Sample-weighted loss compared against a mean of batch means |
| HTTP API | Health, prediction, missing file, incorrect content type, undecodable upload, oversized upload, startup failure |

Continuous integration runs linting, the test suite, an API smoke test against a live
server, and a container build followed by a request to the running container.

---

## Design decisions

**Content-based hashing rather than a seeded shuffle.** A hash is a function of the item
alone, so splits remain stable as the dataset grows. A seeded shuffle reassigns every item
when one is added, invalidating comparison with earlier runs. The cost is that class
balance is approximate; exact stratification would require ranking within each class,
making an item's split depend on the rest of the dataset.

**The checkpoint determines its own architecture.** Architecture parameters are read from
the checkpoint rather than from configuration, with a warning on mismatch. The alternative
allows an edit to a configuration file to invalidate weights on disk, which surfaces only
as shape errors at load time.

**Evaluation runs through the serving code path.** Offline metrics and served predictions
are produced by the same forward pass, so the two cannot diverge silently.

**MLflow is an optional dependency.** Experiment tracking is required for training but not
for serving; excluding it from the core installation keeps Flask, SQLAlchemy, Alembic,
Gunicorn and the Docker SDK out of the inference image.

**Validation findings are separated by severity.** Duplicate images within a single split
over-weight those images without invalidating a measurement, and are reported as warnings.
Duplicates across splits invalidate the measurement and halt the pipeline.

---

## Limitations

- The benchmark is close to saturated. Images are 227×227 centred crops with balanced
  classes, which is not representative of field inspection imagery.
- The backbone is frozen because no GPU was available. Full fine-tuning would be expected
  to improve accuracy.
- The reported confidence is a softmax output and is not calibrated. Since the review
  policy thresholds on it, temperature scaling on the validation split would be the
  appropriate next step.
- Only one dataset is used, so cross-dataset generalisation is not measured.
- 1,598 duplicate images remain within the source dataset. These are reported as warnings
  and do not cross split boundaries.
- The service has no authentication, rate limiting or request persistence, and has not
  been operated under load.

Out of scope by design: batch inference, drift monitoring, a model registry, queueing,
orchestration, and further hyperparameter search.

---

## Repository structure

```
configs/              Hydra configuration groups: data, model, trainer, mlflow, review, service
docker/               Dockerfile.api
docs/                 architecture.md · CODE_TOUR.md · INTERVIEW_NOTES.md · roadmap.md
sentinelinspect/
  config/             Typed configuration schema and loader (Pydantic over Hydra)
  data/               Manifest construction, splitting, validation, Lightning datamodule
  evaluation/         Evaluation entrypoint, metrics, report bundles
  explainability/     Grad-CAM and SHAP attribution
  inference/          Prediction contract, model loader, Predictor, CLI
  inference_service/  FastAPI application, routes, schemas, dependencies, logging
  models/             Shared base module, ResNet-50, ViT, model registry
  preprocessing/      Transform pipeline shared by training and serving
  training/           Training entrypoint
tests/                Unit and integration tests
```

---

## Further documentation

| Document | Contents |
| --- | --- |
| [`docs/architecture.md`](docs/architecture.md) | System boundaries and the trade-off accepted at each |
| [`docs/CODE_TOUR.md`](docs/CODE_TOUR.md) | File-by-file walkthrough in execution order |
| [`docs/roadmap.md`](docs/roadmap.md) | Delivered scope, excluded scope, and defects found during development |
| [`docs/INTERVIEW_NOTES.md`](docs/INTERVIEW_NOTES.md) | Design rationale and known limitations in discussion form |

---

## License

MIT.
