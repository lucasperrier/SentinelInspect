# SentinelInspect

**A visual inspection triage system: reproducible data contracts, a shared inference core, and a deployable API.**

[![CI](https://github.com/lucasperrier/SentinelInspect/actions/workflows/ci.yaml/badge.svg)](https://github.com/lucasperrier/SentinelInspect/actions/workflows/ci.yaml)
![Python](https://img.shields.io/badge/python-3.11-blue)
![License](https://img.shields.io/badge/license-MIT-green)

SentinelInspect classifies concrete surface images as `crack` or `no_crack`, attaches a
confidence score, and **routes uncertain cases to human review** instead of forcing a
decision.

The classifier is the least interesting part. The project exists to demonstrate the
engineering around a model: a dataset that is a versioned, validated contract; one
inference core shared by the CLI, offline evaluation and the HTTP API so they cannot
disagree; tests aimed at failure modes; and a container that is reproducible from a
clean clone.

---

## Architecture

```
data/raw/**  ──build_manifest──▶  manifest.csv  ──splitters──▶  train/val/test.csv
                                       │                              │
                                       └────── validate_dataset ──────┘
                                                     │  fails on leakage
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

**Three properties hold this together.**

1. **The dataset is a contract, not a directory.** Training never walks `data/raw/`. It
   reads `train.csv`, a file you can commit, diff and hand to someone else. Adding images
   tomorrow does not silently change what the model trained on.

2. **Splits are keyed on image content.** `assign_split(sha256)` is a pure function — no
   RNG. The same image lands in the same split on any machine, forever, and byte-identical
   duplicates can never straddle the train/test boundary. Adding 5,000 images does not
   reshuffle the existing ones.

3. **One inference core.** Every prediction — CLI, evaluation, HTTP — goes through
   `Predictor._probabilities`, using the same preprocessing pipeline as training. Three
   thin adapters over one core cannot drift apart.

Detail in [`docs/architecture.md`](docs/architecture.md); a file-by-file walkthrough in
[`docs/CODE_TOUR.md`](docs/CODE_TOUR.md); design rationale and known limitations in
[`docs/INTERVIEW_NOTES.md`](docs/INTERVIEW_NOTES.md).

---

## Quickstart

```bash
git clone https://github.com/lucasperrier/SentinelInspect.git
cd SentinelInspect
python -m venv .venv && source .venv/bin/activate

# CPU-only machine? Install torch from the CPU index first: the default Linux
# wheel pulls several GB of CUDA libraries you will never execute.
pip install torch --index-url https://download.pytorch.org/whl/cpu

pip install -e ".[dev]"
pytest                       # 99 tests, no dataset required
```

The suite builds its own tiny checkpoint, so it needs neither the dataset nor a trained
model.

### The full pipeline

```bash
# 1. inventory data/raw/ -- paths, labels, dimensions, SHA256 per image
sentinelinspect-manifest

# 2. deterministic splits, keyed on content
sentinelinspect-split

# 3. prove the invariants: files readable, no duplicates across splits, no label conflicts
sentinelinspect-validate \
  --manifest data/processed/manifests/manifest.csv \
  --train data/processed/splits/train.csv \
  --val   data/processed/splits/val.csv \
  --test  data/processed/splits/test.csv \
  --raw-root data/raw

# 4. train (Hydra-configured, MLflow-tracked) -- needs the train extra
pip install -e ".[train]"
sentinelinspect-train model.freeze_backbone=true trainer.max_epochs=3

# 5. evaluate a checkpoint into a report bundle
sentinelinspect-evaluate "checkpoint_path='runs/<run>/<file>.ckpt'" split=test

# 6. classify one image
sentinelinspect-predict \
  "checkpoint_path='runs/<run>/<file>.ckpt'" \
  "image_path='data/raw/ccic/Positive/00001.jpg'"
```

> Quote the Hydra overrides. Checkpoint filenames contain `=` (`epoch=00`), which the
> override parser otherwise reads as a separator.

Swap the model with a config group — no code change:

```bash
sentinelinspect-train model=vit
```

---

## The API

```bash
pip install -e ".[api]"
SENTINELINSPECT_CHECKPOINT=runs/<run>/<file>.ckpt \
  uvicorn sentinelinspect.inference_service.app:app --port 8000
```

The model loads during startup, before the first request is accepted. A missing or
unreadable checkpoint crashes the process rather than letting it report healthy and fail
every call.

### `GET /health`

```json
{
  "status": "ok",
  "model_loaded": true,
  "model_name": "resnet50",
  "checkpoint_sha256": "3f2a8c1d9e4b7a05",
  "package_version": "0.1.0",
  "review_band": [0.35, 0.65]
}
```

The checkpoint fingerprint is there so you can tell two deployments apart. "Is it up?" is
rarely the question you have during an incident; "which weights are up?" is.

### `POST /predict`

```bash
curl -F "file=@image.jpg;type=image/jpeg" http://localhost:8000/predict
```

```json
{
  "predicted_label": "crack",
  "predicted_index": 1,
  "confidence_score": 0.9731,
  "probabilities": { "no_crack": 0.0269, "crack": 0.9731 },
  "needs_review": false,
  "review_reason": null,
  "model_metadata": {
    "name": "resnet50",
    "backbone": "resnet50",
    "checkpoint_path": "runs/.../model.ckpt",
    "checkpoint_sha256": "3f2a8c1d9e4b7a05",
    "package_version": "0.1.0",
    "class_names": ["no_crack", "crack"]
  },
  "latency_ms": 41.2
}
```

This is the same `Prediction` object the CLI prints and offline evaluation produces — one
Pydantic definition, not three.

| Status | Cause |
| --- | --- |
| `200` | Prediction returned |
| `400` | Empty upload, or bytes that are not a decodable image |
| `413` | Upload exceeds `SENTINELINSPECT_MAX_UPLOAD_MB` (default 10) |
| `415` | Content type is not `image/*` |
| `422` | No file field in the request |
| `503` | Model not loaded |

Interactive docs at `/docs`.

### Container

```bash
docker build -f docker/Dockerfile.api -t sentinelinspect-api .
docker run --rm -p 8000:8000 \
  -v "$(pwd)/runs:/models:ro" \
  -e SENTINELINSPECT_CHECKPOINT=/models/<run>/<file>.ckpt \
  sentinelinspect-api
```

Multi-stage, non-root, CPU-only torch, `HEALTHCHECK` on `/health`. Weights are mounted
rather than baked in, so a new checkpoint does not mean a new image.

---

## Triage: the `needs_review` rule

A prediction is routed to a human when `p(crack)` falls inside a confidence band:

```yaml
# configs/review/default.yaml
lower: 0.35
upper: 0.65
```

The band is **two-sided**. A one-sided confidence floor would flag `p=0.48` and wave
through `p=0.52`, though both are equally uncertain.

It is tuned on the validation split under a review-capacity budget, not guessed —
`evaluation.metrics.tune_review_band` finds the narrowest band catching a target share of
the model's errors while flagging no more than a set fraction of traffic. Unconstrained,
"catch more errors" always widens to `[0, 1]`.

---

## Results

ResNet-50 with a **frozen backbone** (linear probe), 3 epochs on CPU, 28,054 training
images. Evaluated once on a 5,889-image test split containing no image the model has seen.

| Metric | Test | Validation |
| --- | --- | --- |
| Accuracy | **0.9885** | 0.9846 |
| F1 | 0.9886 | 0.9844 |
| ROC AUC | 0.9990 | 0.9987 |
| Recall (crack) | 0.9899 | 0.9855 |
| Loss | 0.0707 | 0.0741 |

```
confusion (test)   tn=2872   fp=38   fn=30   tp=2949
```

### What triage buys

With the band tuned on validation and applied unchanged to test:

| | |
| --- | --- |
| Routed to review | 383 of 5,889 — **6.5%** |
| Model errors intercepted | 54 of 68 — **79.4%** |
| Accuracy on the 5,506 auto-decided | **0.9975** |

The band was tuned for 80% error recall on validation and delivered 79.4% on test, so it
generalised rather than fitting the tuning split. Sending 6.5% of images to a human lifts
accuracy on everything decided automatically from 98.85% to 99.75%.

> **On the earlier 99.78%.** A previous version of this repository reported that figure. It
> was measured on a test set where 8,482 images had also been in training, because the
> dataset directory contained a byte-identical duplicate of itself under a second name and
> splits were keyed on filename. The number above is lower and real. See
> [`docs/INTERVIEW_NOTES.md`](docs/INTERVIEW_NOTES.md).

**Honest caveats.** CCIC is a near-saturated benchmark — 227x227 centred crops, perfectly
balanced. A frozen backbone was chosen because no GPU was available; a full fine-tune would
score higher. The confidence is a softmax output, not a calibrated probability. Single
dataset, so nothing here measures cross-dataset generalisation.

---

## Testing

```bash
pytest                       # 99 tests in ~7 seconds
```

Aimed at failure modes rather than line coverage:

| Area | What is pinned |
| --- | --- |
| Splitting | Determinism; independence from row order; **stability when the dataset grows**; byte-identical images never separated |
| Validation | Content overlap the path check cannot see; warnings vs errors; label conflicts |
| Contract | Review-band edges; inverted bands rejected; label/index consistency |
| Predictor | Path, bytes, PIL and array inputs all agree; image and tensor paths agree; corrupt input raises a typed error; the model loads once |
| Metrics | Sample-weighted loss vs mean-of-batch-means, with the numbers that made it wrong |
| API | Health, prediction, missing file, wrong content type, corrupt upload, oversized upload, startup failure |

Two tests exist specifically to keep others honest: one pins the *old* path-keyed
behaviour, so the leakage regression test cannot pass vacuously on a fixture with no
duplicates; and one compares an HTTP response against the app's own `Predictor` instance,
so "shared inference core" is verified rather than asserted.

---

## Design decisions

**Hashing instead of `train_test_split(random_state=42)`.** A hash is a pure function of
the item, so the split is stable under dataset growth. The cost is that class balance is
approximate rather than exact. Exact stratification requires ranking within each class,
which makes an item's split depend on every other item — stability and exactness are
mutually exclusive, and stability is worth more.

**The checkpoint outranks the config on architecture.** A `.ckpt` knows what it is.
Letting `configs/model/*.yaml` win means editing a YAML silently invalidates weights on
disk, and you find out through shape-mismatch errors at deploy time.

**Evaluation runs through the Predictor.** Otherwise a reported metric and a served
decision are computed by different code. This project previously had a torchvision
transform in the CLI and albumentations everywhere else; the measured divergence was
0.035 max tensor delta — small, uncontrolled, and growing.

**MLflow is not a core dependency.** It is a training concern. Keeping it in an extra
keeps Flask, SQLAlchemy, alembic, gunicorn, graphene and the Docker SDK out of the
inference container.

**Warnings and errors are different.** Duplicate images *within* one split over-weight
that image; duplicates *across* splits invalidate the measurement. The first is a warning,
the second stops the pipeline.

---

## Deliberately out of scope

Batch inference, drift monitoring, a model registry, authentication, a queue, a database,
Kubernetes, ONNX export, and further hyperparameter search. The scope was one week and
fixed in advance. Six things that work beat twelve that half-work.

---

## Repository layout

```
configs/           Hydra config groups: data, model, trainer, mlflow, review, service
docker/            Dockerfile.api
docs/              architecture.md · CODE_TOUR.md · INTERVIEW_NOTES.md · roadmap.md
sentinelinspect/
  config/          typed schema + loader (Pydantic over Hydra)
  data/            manifest · splitters · validation · datamodule
  evaluation/      evaluate · metrics · reports
  explainability/  Grad-CAM · SHAP
  inference/       contracts · model_loader · predictor · predict (CLI)
  inference_service/  FastAPI app · routes · schemas · dependencies · logging
  models/          base · resnet50 · vit · factory
  preprocessing/   the transform pipeline training and serving share
  training/        train.py
tests/             unit + integration
```
