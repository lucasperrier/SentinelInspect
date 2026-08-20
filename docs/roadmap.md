# Roadmap

## SentinelInspect — ship the strongest version of what this repository already is

This roadmap is written against the **current state of the repo**, not an idealized
platform. The order is strict: each phase assumes the previous one is finished cleanly.

The governing product question:

> Can this repository show that a CV prototype was turned into a controlled, testable,
> deployable inspection workflow instead of stopping at training and offline metrics?

That is the hiring signal this roadmap optimizes for.

---

## Scope principle

The repo gets stronger by finishing a **small number of visible engineering surfaces well**,
not by pretending to be a full platform before the core workflow runs. Scope stays divided
into **must ship**, **nice to have**, and **defer**.

---

## Current state (accurate to the code)

**Working:** the data layer (`build_manifest.py`, `splitters.py`, `validate_dataset.py`,
`datamodule.py`), preprocessing (`transforms.py`), both model modules (`resnet50.py`,
`vit.py`), the Hydra+MLflow training entrypoint, the offline evaluation script, single-image
inference, Grad-CAM/SHAP explainability, typed config validation, and the data-layer tests.

**Written but not yet green end-to-end:** `train.py` and `evaluate.py` still call
`CrackDataModule` with the pre-refactor argument names, so the training path does not run as-is.

**Empty scaffolding:** the inference service, batch/shared inference, evaluation helper
modules, monitoring, MLOps, jobs, utils, the Dockerfiles, CI, and `pyproject.toml`.

---

## Phase 0 — Make the core path executable

**Must ship. Do this first.**

### Why first

The data layer was recently refactored so persisted manifests and splits are the source of
truth, but the training and evaluation entrypoints have not caught up. Until a clean
end-to-end run exists, every downstream phase rests on unverified code, and a reviewer who
clones the repo cannot run anything.

### Tasks

- align the `CrackDataModule` constructor call in `sentinelinspect/training/train.py` and
  `sentinelinspect/evaluation/evaluate.py` with the refactored datamodule signature. The datamodule now
  takes `train_split_path` / `val_split_path` / `test_split_path` / `robustness_split_path`,
  **not** the old `val_split` / `test_split` / `robustness_split` float arguments.
- add a single, well-defined label-encoding step. `build_manifest.py` writes string labels
  (`crack` / `non_crack`), but `datamodule._df_to_dataset` casts `label` with `int(...)` and
  asserts values are `0` / `1`. Pick one place — the splitter output, a manifest column, or
  the datamodule — to own the string-to-integer mapping so the contract is explicit.
- add a tiny committed fixture dataset (a dozen images per class) or a documented download
  script so the pipeline can be exercised without external data.
- run `manifest -> splits -> validate -> train (1 epoch, CPU) -> evaluate -> predict` once and
  commit the resulting `reports/` bundle.

### Success criteria

- `python -m sentinelinspect.training.train` completes on CPU against the fixture data
- `python -m sentinelinspect.evaluation.evaluate` writes a full `reports/` bundle
- a fresh clone can reproduce a run from the README alone

---

## Phase 1 — Data artifacts as the source of truth

**Must ship. Largely done.**

### Current state

Persisted manifest and split CSVs are already the canonical dataset contract, consumed by
`sentinelinspect/data/datamodule.py`. The earlier version that scanned raw folders and re-split at runtime
has been refactored out.

### Tasks

- route any remaining hard-coded dataset paths through config rather than literals
- decide whether `robustness.csv` (currently a full copy of the manifest) is a true held-out
  evaluation split or a later extension, and document the choice
- confirm `validate_dataset.py` can block downstream training/evaluation when artifacts are
  invalid (duplicates, missing/corrupt files, or split overlap)

### Success criteria

- training performs no hidden runtime re-splitting
- evaluation uses the same saved splits as training
- manifest + split CSVs are enough to reconstruct dataset membership for an experiment
- invalid manifests or overlapping splits fail fast

---

## Phase 2 — Standardize the evaluation bundle

**Must ship.**

### Why

The metric logic already exists in `evaluate.py` (accuracy, F1, ROC AUC, confusion matrix,
classification report, `predictions.npz`). The gap is that the contract is not yet
standardized or confidence-aware, and `sentinelinspect/evaluation/metrics.py` is still empty.

### Tasks

- move reusable metric helpers out of `evaluate.py` into `sentinelinspect/evaluation/metrics.py`
- keep `evaluate.py` as the entrypoint but stabilize its output into a fixed bundle
- save positive-class confidence scores per sample in a reusable form
- add a first-pass triage rule: mark predictions `needs_review` when confidence falls inside a
  configurable low-confidence band
- include a simple threshold / coverage-vs-risk summary so the review threshold has
  quantitative backing
- (optional, supportive) wire the existing `sentinelinspect/evaluation/robustness.py` IoU and faithfulness
  helpers into the bundle rather than leaving them standalone

### Success criteria

- every evaluated model produces the same artifact structure in `reports/`
- confidence scores are saved in a reusable form
- `needs_review` is derivable directly from saved predictions
- evaluation outputs can support a deployment decision, not just serve as historical logs

---

## Phase 3 — Shared inference and a usable API

**Must ship.** This is the surface that most directly backs the "deployable" claim, so it
precedes monitoring.

### Why

There is already a working single-image path in `sentinelinspect/inference/predict.py` (Hydra config,
checkpoint load, preprocess, softmax, predicted class). The shortest path is to extract and
reuse that logic, not rebuild it. Note that `predict.py` currently builds its own
`torchvision` transform while training/eval use `albumentations`; the shared core should
remove that train/serve skew.

### Tasks

- extract a single shared prediction core (e.g. `predict_one(image, model) -> result`)
- have `predict.py`, `batch_predict.py`, and the FastAPI service all call that core
- finish `sentinelinspect/inference/batch_predict.py`
- complete the service layer in `sentinelinspect/inference_service/`
  (`schemas.py`, `routes.py`, `app.py`, `dependencies.py`)
- return the full output contract on every request:
  `predicted_label`, `confidence_score`, `needs_review`, `model_metadata`

### Success criteria

- single-image, batch, and API inference share one code path
- API and batch outputs are numerically consistent on the same image
- the service starts locally and passes a smoke test
- request/response schemas make the interface clear from the repo alone

---

## Phase 4 — Real packaging, CI, and Docker

**Must ship.**

### Why

The clearest unfinished delivery surfaces are all visible at the top level, so finishing them
has outsized hiring value: `pyproject.toml` is empty, `.github/workflows/ci.yaml` is empty,
and both Dockerfiles are empty.

### Tasks

- populate `pyproject.toml` with project metadata and dependencies, aligned with
  `requirements.txt`
- turn `.github/workflows/ci.yaml` into a working GitHub Actions workflow with, at minimum:
  ruff lint, unit tests, and the Phase 0 end-to-end smoke test
- implement `docker/Dockerfile.train`
- implement `docker/Dockerfile.api`

### Success criteria

- the repository installs reproducibly
- pull requests run automated checks
- train and API images build successfully
- core code paths are protected by CI instead of manual trust

---

## Phase 5 — Minimal monitoring hooks

**Nice to have. Keep intentionally small** — this should support the triage story, not become
an observability platform.

### Tasks

- implement structured prediction logging in `sentinelinspect/monitoring/prediction_logger.py`, with the
  minimum useful fields: timestamp, model identifier, predicted label, confidence, `needs_review`
- add one small offline report in `sentinelinspect/monitoring/reporting.py`
- keep any drift analysis in `sentinelinspect/monitoring/drift.py` lightweight and honest

### Success criteria

- inference outputs can be logged consistently
- one offline summary can compare reference vs. recent confidence distributions
- the repo demonstrates post-deployment thinking without overstating platform maturity

---

## Deferred

Out of core scope until the must-ship phases are cleanly finished:

- registry/promotion workflows in `sentinelinspect/mlops/`
- rollback automation
- champion/challenger orchestration
- extensive OOD benchmarking
- active learning, pseudo-labeling, distillation
- cloud deployment buildout / Kubernetes
- ONNX or TorchScript export, unless latency profiling proves it matters

These are fine future directions but should not compete with the strongest v1 story.

---

## Recommended implementation order

1. make the core path executable (Phase 0)
2. confirm data artifacts as source of truth (Phase 1)
3. standardize the evaluation bundle (Phase 2)
4. extract shared inference and finish the service (Phase 3)
5. complete packaging, CI, and Docker (Phase 4)
6. add minimal monitoring hooks (Phase 5)

---

## Hiring signal this roadmap is optimized for

When v1 is shipped, the project should credibly support:

- built a production-first CV triage system rather than a notebook-only classifier
- implemented deterministic dataset lineage through manifests and persisted split artifacts
- standardized evaluation outputs with confidence-aware triage analysis
- shipped shared inference across single-image, batch, and API workflows
- added reproducible packaging, CI, and Dockerized execution