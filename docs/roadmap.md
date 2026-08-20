# Roadmap — what was planned, and what shipped

This began as a forward-looking plan. It is now a record, because every phase has either
landed or been deliberately cut. Kept rather than deleted: what a project chose *not* to
build is as informative as what it did.

Scope was fixed in advance at one week, with a cut list agreed before starting.

---

## Delivered

| Phase | Outcome |
| --- | --- |
| **0 · Data integrity** | Splits keyed on `sha256`; content-level leakage detection; errors and warnings separated. Closed a leak putting 8,482 images in both train and test. |
| **1 · Artifacts as source of truth** | The manifest and split CSVs are the contract. Absolute paths removed, so the contract survives a directory rename. |
| **2 · Standardised evaluation** | `metrics.py` and `reports.py` extracted; sample-weighted loss; per-sample confidence in the bundle; a tuned `needs_review` band. |
| **3 · Shared inference core** | `Predictor` behind CLI, offline evaluation and FastAPI. Typed contract with `needs_review` and model provenance. |
| **4 · Packaging, CI, Docker** | Installable package with console scripts; GitHub Actions running lint, tests, an API smoke test and a container build; multi-stage CPU-only image. |

---

## Cut, and why

| Item | Reason |
| --- | --- |
| Batch-inference CLI | On the plan's "cut first if behind" list. `Predictor.predict_images` covers the need without a second entrypoint. |
| Structured prediction logging beyond request logs | Same list. The API logs one JSON line per request; persisting predictions needs storage, which needs a retention decision. |
| Drift monitoring, model registry, active learning | On the "do not build" list. Each is a project, not a feature. |
| Kubernetes, cloud deployment, auth, queues, a database | Same. The container is the deployment boundary this project claims. |
| ONNX export, a third architecture, more HPO | Same. None would change what the repository demonstrates. |
| Real SDNET2018 | Discovered late that both archives on disk were the *same* dataset. Adding a genuinely different source is the right next step, and is honest work not yet done. |

---

## Found along the way, and fixed

None of these were on the plan. All were discovered by running the code rather than reading it.

1. **`data/raw/sdnet2018/` was a byte-identical copy of `data/raw/ccic/`** — 8,482 test
   images had been trained on, and the path-based validator reported PASSED.
2. **The `preprocessing:` config block had never affected a run** — a flat-versus-nested
   dict mismatch, silently falling back to defaults.
3. **`stratify_by` was a no-op** — grouping by label produced output identical to not
   grouping, because the score never depended on the group.
4. **The explainability entrypoint had been broken for five months** — constructed the
   datamodule with pre-refactor arguments. Underneath: an undeclared `scikit-image`
   dependency meaning SHAP had never run, a hardcoded ViT patch grid, and a KernelSHAP
   unpacking assumption that NumPy 2 no longer tolerates.
5. **`evaluate.py` scored the test set twice** and reported a mean of batch means as the loss.
6. **Renaming the project directory broke the dataset** — absolute paths in the manifest.
7. **Console scripts failed outside the repository root** — Hydra's relative `config_path`
   does not survive an installed entry point.
8. **The API image installed Flask, SQLAlchemy, alembic and the Docker SDK** — transitive
   from MLflow, which serving never uses.

---

## If there were a second week

In priority order:

1. **Real SDNET2018.** Roughly 8% positive across three substrates — the honest stress
   test, and enough of a class imbalance to make the review band earn its keep.
2. **Calibration.** `confidence_score` is a softmax output, not a calibrated probability.
   Temperature scaling on validation would make the number mean what it appears to mean,
   which matters because the triage rule thresholds on it.
3. **Prediction logging.** So the review rate can be tracked over time rather than measured
   once at evaluation.
4. **A full fine-tune on a GPU.** The current model is a linear probe on frozen ImageNet
   features, chosen because this machine has no GPU.
