# Code tour

A file-by-file walkthrough of how this repository actually executes, written in
execution order rather than directory order. Session 1 follows a training run
end to end. Session 2 covers evaluation, inference and explainability.

---

## Session 1 — anatomy of a training run

```bash
python -m sentinelinspect.training.train
```

Nine things happen. Each one is a boundary worth understanding.

---

### 1 · Python resolves the import

`-m src.training.train` works only because the current directory is on
`sys.path` and you are standing in the repo root. There are no `__init__.py`
files and `pyproject.toml` is empty, so `src` is not an installed package — it
is a directory that happens to be importable from one specific location.
`tests/conftest.py` papers over the same gap for pytest by inserting the repo
root into `sys.path`.

This is why moving a script one directory away breaks every import. Stage 2
fixes it with a real editable install.

---

### 2 · Hydra composes the configuration

```python
@hydra.main(version_base=None, config_path="../../configs", config_name="train")
def main(cfg: DictConfig) -> None:
```

The decorator runs *before* `main` gets called. It reads `configs/train.yaml`,
whose `defaults` list names one file from each **config group**:

```yaml
defaults:
  - data: default        # configs/data/default.yaml    -> cfg.data
  - model: resnet50      # configs/model/resnet50.yaml  -> cfg.model
  - trainer: default     # configs/trainer/default.yaml -> cfg.trainer
  - mlflow: default      # configs/mlflow/default.yaml  -> cfg.mlflow
  - _self_               # this file's own keys, merged LAST
```

Each group becomes a key in one merged tree. `_self_` sitting last means keys
written directly in `train.yaml` override anything the groups set.

**Swapping a group** is the point of the design:

```bash
python -m sentinelinspect.training.train model=vit          # loads configs/model/vit.yaml instead
python -m sentinelinspect.training.train trainer.max_epochs=3
python -m sentinelinspect.training.train "checkpoint_path='runs/a-epoch=03.ckpt'"
```

The quoting in the third line is not decoration. Hydra's override parser treats
`=` as a separator, and checkpoint filenames contain `epoch=03`, so an unquoted
path is a syntax error. This is why the README's inference command does not run
as written.

`${model.name}` inside `run_name` is OmegaConf **interpolation** — resolved at
read time, not at write time.

**Side effects of the decorator.** It creates `outputs/<date>/<time>/`
containing a `.hydra/` snapshot of the exact merged config and a log file.
Every run is therefore self-documenting. It does *not* change the working
directory in this version (hydra-core 1.3.5, `version_base=None`) — verified
by experiment — which is why relative paths like `data/processed/...` resolve.
Do not rely on that silently: under older Hydra defaults it *did* chdir, and
every relative path in the config would break.

---

### 3 · Pydantic validates the tree

```python
runtime = to_runtime_config(cfg)      # sentinelinspect/config/load.py
```

`OmegaConf.to_container(cfg, resolve=True)` turns the config into a plain dict
with interpolations expanded, then `RuntimeConfig.model_validate(raw)` type-checks
it against `sentinelinspect/config/schema.py`.

Two `ConfigDict` settings do most of the work:

- `extra="forbid"` (DataConfig, TrainerConfig, MlflowConfig, ServiceConfig) —
  an unknown key is an error. This catches typos: `bacth_size: 64` fails loudly
  instead of being silently ignored.
- `extra="allow"` (ModelConfig, PreprocessingConfig, RuntimeConfig) — extra keys
  pass through. Necessary because ViT carries `drop_rate`, `drop_path_rate` and
  `label_smoothing` that ResNet does not.

`Field(gt=0)` and friends express constraints the type system cannot:
`num_classes: int = Field(gt=1)`, `learning_rate: float = Field(gt=0.0)`.

`@model_validator(mode="after")` expresses constraints spanning several fields —
"trainer is required for train/eval", "split is required for eval". A per-field
validator cannot see other fields; this one runs once the whole object exists.

**The gap.** `runtime` is validated, but the very next lines use the *unvalidated*
dict:

```python
cfg_dict = OmegaConf.to_container(cfg, resolve=True)
model = build_model(cfg_dict["model"])          # raw dict, not runtime.model
```

So validation currently acts as an assertion that runs and is then largely
ignored on the model path. Worth closing.

---

### 4 · Seeding

```python
seed_everything(runtime.seed, workers=True)
```

Seeds Python's `random`, NumPy and Torch. `workers=True` also seeds each
DataLoader worker process, which otherwise inherit an unseeded state and make
augmentation non-reproducible.

Note this has nothing to do with the train/val/test split — that is decided by
hashing in `splitters.py` and never touches an RNG at all.

---

### 5 · MLflow opens a run

```python
mlflow.set_tracking_uri("sqlite:///mlflow.db")
with mlflow.start_run(run_name=run_name) as run:
    mlflow.log_params(flatten_dict(cfg_dict))
```

`flatten_dict` turns the nested config into dotted keys (`model.learning_rate`)
because MLflow params are flat key/value pairs.

```python
logger = MLFlowLogger(..., run_id=run.info.run_id)
```

Passing `run_id` attaches Lightning's logger to the run already open, rather
than starting a second one. Without it you would get two runs per training job,
one with params and one with metrics.

---

### 6 · The datamodule is constructed

`CrackDataModule` subclasses `pl.LightningDataModule`. Lightning calls its
methods in a fixed order — this is a **template method** pattern: the base class
owns the sequence, the subclass fills in the steps.

- **`prepare_data()`** — called once, on one process. Meant for downloads. Here
  it only asserts the three split CSVs and the manifest exist.
- **`setup(stage)`** — called on *every* process, once per stage. This is where
  the datasets are built, because each worker needs its own copy.

`setup` runs the validator as a gate:

```python
report = validate(...)
if report.errors and self.fail_on_validation_error:
    raise ValueError(...)
```

Then `_df_to_dataset` reads each split CSV and maps the string label to an
integer (`crack` -> 1, `non_crack` -> 0), raising on anything outside that
vocabulary. `CrackDataset.__getitem__` opens the image with PIL, converts to
RGB, converts to a NumPy array (because albumentations wants arrays, not PIL
images), applies the transform, and returns `(tensor, label)`.

**A live defect on this path.** `train.py` passes
`preprocessing={"image_size": ...}` — a *flat* dict. `transforms.py` does
`cfg.get("preprocessing", {})`, expecting a *nested* one. It finds nothing and
silently uses hardcoded defaults, so the entire `preprocessing:` block in
`configs/train.yaml` has never affected a run. `tests/unit/test_preprocessing.py`
passes because it calls the function with a correctly nested dict — a unit test
that verifies the contract while nobody upholds it.

---

### 7 · The model is built

```python
def build_model(model_cfg):
    name = str(model_cfg.get("name", "resnet50")).lower()
    if "vit" in name:
        return VisionTransformerModule(model_cfg)
    return ResNet50Module(model_cfg)
```

This function exists in **three** copies — `train.py`, `evaluate.py` and
`predict.py` — while `sentinelinspect/models/factory.py`, which is exactly where it belongs,
is an empty file.

Inside `ResNet50Module.__init__`:

- `save_hyperparameters(config)` stores the dict as `self.hparams`, an
  `AttributeDict` supporting both `self.hparams.learning_rate` and
  `self.hparams.get("x", default)`. Lightning saves it into the checkpoint, which
  is how `load_from_checkpoint` can rebuild the architecture later.
- `timm.create_model(model_name, pretrained=..., num_classes=2)` builds the
  backbone and replaces the 1000-class ImageNet head with a 2-class one.
- **Nine separate metric objects** are created: accuracy, F1 and AUROC for each
  of train, val and test. They are separate because torchmetrics objects
  *accumulate state* across batches. One shared accuracy object would mix
  training and validation batches into a single meaningless number.

`_shared_step` is the real work, called by all three of `training_step`,
`validation_step` and `test_step` — another template method, this time
hand-rolled. It computes logits, loss, `preds` (argmax) and `probs` (softmax
column 1), feeds `preds` to accuracy/F1 and `probs` to AUROC — AUROC needs a
continuous score, not a hard decision — then logs.

`self.log(name, metric_object)` passes the *object*, not a number. Lightning
knows how to call `.compute()` at epoch end and `.reset()` afterwards.

Two blemishes here: `metric_factory` at `resnet50.py:28` is assigned and never
called — dead code. And `_shared_step` in `resnet50.py` uses
`if/elif/elif` with no `else`, so an unexpected stage name raises
`UnboundLocalError`; `vit.py` uses `else` and is safe. The two files should
share one base class.

`configure_optimizers` is a Lightning hook: return an optimizer (or a dict with
a scheduler) and Lightning wires up the step calls.

---

### 8 · `trainer.fit(model, datamodule=datamodule)`

Lightning now drives, calling in order: `prepare_data` -> `setup("fit")` ->
`configure_optimizers` -> `train_dataloader()` / `val_dataloader()` -> a short
sanity-check validation pass -> the epoch loop.

Two callbacks are attached:

- `ModelCheckpoint(monitor="val_loss", mode="min", save_top_k=1)` — writes the
  single best checkpoint to `runs/<experiment>-<run_name>/`. It knows about
  `val_loss` only because `_shared_step` logged it under that name.
- `LearningRateMonitor` — records the LR each epoch.

---

### 9 · Export and final metrics

The best checkpoint is copied to `trained_models/<experiment>/<timestamp>/` and
logged as an MLflow artifact, then `trainer.validate` and `trainer.test` run with
`ckpt_path=best`, reloading the best weights rather than the last ones.

---

## Where things land on disk

| Path | Written by | In git? |
| --- | --- | --- |
| `data/processed/manifests/` | `build_manifest.py` | no |
| `data/processed/splits/` | `splitters.py` | no |
| `outputs/<date>/<time>/` | Hydra, every run | no |
| `mlflow.db` | MLflow tracking store | no |
| `runs/<exp>-<run>/` | `ModelCheckpoint` | no (`**/*.ckpt`) |
| `trained_models/` | `train.py` export step | no |
| `reports/eval/` | `evaluate.py` | no |

Nothing a run produces is committed. The repository holds code and
configuration; every artifact is reproducible from them.

---

## The OOP in this repository

| Pattern | Where | Why |
| --- | --- | --- |
| Template method | `LightningModule`, `LightningDataModule` | The framework owns the sequence; you fill in steps. You never call `training_step` yourself. |
| Template method, hand-rolled | `_shared_step` | Three public hooks delegate to one implementation. |
| Inheritance | `CrackDataModule(pl.LightningDataModule)` | Correct here: Lightning dispatches on the base type. |
| Composition | `self.model = timm.create_model(...)` | The module *has a* backbone rather than *being one*. Lets you swap ResNet for ViT without touching the training logic. |
| `@dataclass` | `ManifestRecord`, `ValidationReport` | Plain data, no validation needed. Free `__init__` and `__repr__`. |
| Pydantic `BaseModel` | `sentinelinspect/config/schema.py` | Data crossing a trust boundary — validates and coerces types at the edge. |
| Factory function | `build_model` | Picks a class from a string. Currently duplicated three times. |

**Rule of thumb this codebase illustrates:** inherit when a framework needs to
dispatch on your type; compose when you just need the other object's behaviour.

---

## Defects on the training path, as of Stage 0

1. Preprocessing config is inert (flat vs nested dict) — Section 6.
2. `build_model` duplicated three times; `models/factory.py` empty — Section 7.
3. `metric_factory` dead code at `resnet50.py:28`.
4. `_shared_step` in `resnet50.py` has no `else` branch.
5. Pydantic validation runs but `build_model` uses the raw dict — Section 3.
6. Nothing on this path is tested at all.

---

## Session 2 — evaluation, inference, explainability, and the empty half

Session 1 followed one command all the way down. This session covers the other
three entrypoints and then inventories what does not exist yet.

---

### A · Evaluation — `sentinelinspect/evaluation/evaluate.py`

```bash
python -m sentinelinspect.evaluation.evaluate "checkpoint_path='runs/.../model.ckpt'"
```

Same Hydra + Pydantic front end as training, reading `configs/eval.yaml`. Then:
build the datamodule, load the checkpoint, run the test set, write a bundle of
artifacts to `reports/eval/`:

| File | Contents |
| --- | --- |
| `metrics.json` | accuracy, F1, AUC, confusion matrix, per-class report |
| `classification_report.txt` | the same report as text |
| `confusion_matrix.npy` | the matrix as an array |
| `predictions.npz` | `y_true`, `y_pred`, `y_prob_pos` per sample |

`predictions.npz` is the important one: keeping per-sample outputs means any new
metric can be computed later without re-running the model.

**Defects on this path.**

1. **The test set is scored twice.** `trainer.test(...)` runs a full pass, then
   `predict_on_test(...)` runs a second one to collect raw arrays Lightning did
   not return. Two forward passes over 5,889 images — about 5 minutes of the
   10 that a CPU evaluation takes.
2. **`setup()` is called twice.** Explicitly at `datamodule.setup(stage="test")`,
   then again by `trainer.test(datamodule=...)`. Each call re-runs dataset
   validation over the whole manifest.
3. **`test_loader` is assigned and never used.**
4. **The loss is a mean of batch means.** `np.mean(losses)` averages per-batch
   figures, but the final batch is usually smaller, so it is over-weighted. Correct
   would be to accumulate `loss * len(batch)` and divide by the sample count.
5. **The output directory is hardcoded** to `reports/eval`, so every run
   overwrites the previous one. The historical `reports/eval_trained_vit/` folders
   were renamed by hand.
6. **`split` is validated and ignored.** `configs/eval.yaml` sets `split: test`
   and `RuntimeConfig` constrains it, but the code always calls
   `test_dataloader()`. Evaluating on val is impossible despite looking configurable.
7. **`except Exception` around `roc_auc_score`** sets AUC to `None` on *any*
   failure, including a bug in the calling code. Only the "one class present"
   case deserves catching.
8. **No `needs_review` threshold and no per-sample confidence in the report.**
   `predictions.npz` has the probabilities, but nothing turns them into the
   triage decision the README promises.

---

### B · Inference — `sentinelinspect/inference/predict.py`

After the Stage 1 fixes this is 61 lines: load config, load checkpoint via the
factory, build the *shared* inference transform, run one image, print a dict.

```python
tfm = build_inference_transforms(
    runtime.preprocessing.model_dump() if runtime.preprocessing else None
)
image = Image.open(image_path).convert("RGB")
x = tfm(image=np.array(image))["image"].unsqueeze(0).to(device)
```

`np.array(image)` because albumentations works on arrays; `["image"]` because
albumentations returns a dict; `.unsqueeze(0)` to add the batch dimension a
model always expects.

**What is still missing for a real inference core.**

- The output is a printed dict, not a typed object. `predicted_class: 0` rather
  than `no_crack`, no `needs_review`, no model provenance.
- The model is loaded from disk on every invocation. Fine for a CLI, fatal for a
  service handling requests.
- Nothing is reusable: a batch path or an HTTP route would have to copy this.

That is precisely what Stage 4 replaces with a `Predictor` class.

---

### C · Explainability — `sentinelinspect/explainability/`

836 lines, the largest subsystem in the repo, and entirely disconnected from
the other three entrypoints.

**`grad_cam.py` (157 lines) — the best-written file here.**

Grad-CAM answers "which regions of the image drove this prediction?" It works
by capturing two things during a forward/backward pass using **hooks**:

```python
self._fwd_handle = target_layer.register_forward_hook(self._forward_hook)
self._bwd_handle = target_layer.register_full_backward_hook(self._backward_hook)
```

A hook is a callback PyTorch invokes when data flows through a module. The
forward hook stores the layer's *activations*; the backward hook stores the
*gradients* of the chosen class score with respect to those activations. Neither
is otherwise reachable — they are intermediate values discarded after the pass.

For a CNN the activations are `(B, C, H, W)` and the classic algorithm applies:

```python
weights = grads.mean(dim=(2, 3), keepdim=True)   # (B,C,1,1) how much each channel mattered
cam = (weights * acts).sum(dim=1, keepdim=True)  # (B,1,H,W) weighted sum of feature maps
cam = F.relu(cam)                                # keep only positive evidence
cam = self._normalize(cam)                       # per-sample rescale to [0,1]
```

Averaging the gradient over space gives one importance number per channel; the
weighted sum turns 2048 feature maps into a single relevance map; ReLU discards
evidence *against* the class.

For a ViT the activations are `(B, N, D)` tokens, so there is no spatial grid to
average over. The file handles that separately, dropping the CLS token and
reshaping the remaining patches back into a grid:

```python
if N == 1 + H_p * W_p:      # timm ViTs prepend a CLS token
    acts_p, grads_p = acts[:, 1:, :], grads[:, 1:, :]
token_scores = (acts_p * grads_p).sum(dim=2)   # (B, H_p*W_p)
cam = token_scores.view(B, 1, H_p, W_p)
```

Hooks stay registered until removed, so `close()` exists and must be called.
There is no context-manager form, which is the one thing this class is missing.

**`shap_utils.py` (288 lines)** — SHAP over superpixels for CNNs and over
patches for ViTs, via `KernelExplainer`. It re-implements ImageNet normalisation
inline rather than importing it.

**`utils.py` (98 lines)** — denormalise, colourise, overlay, save PNG. Defines
`IMAGENET_MEAN` / `IMAGENET_STD` as yet another copy of those constants.

**`run_explainability.py` (293 lines) — orphaned.**

This entrypoint does not use Hydra at all:

```python
def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
```

Plain `yaml.safe_load` plus `argparse`. So the repository contains **two config
systems**, and this one reads files from `configs/old/` — a directory already
scheduled for deletion. It also carries a **fourth** copy of `build_model`,
which dispatches on `cfg["model"]` (the timm backbone name) where the other
three used `cfg["name"]` (the registry key). Different key, same function name.

And `vit_grid_from_name` ignores its own argument:

```python
def vit_grid_from_name(model_name: str) -> Tuple[int, int]:
    return (14, 14)      # correct only for patch16 at 224
```

---

### D · The empty half — 39 files

| Group | Files | Verdict |
| --- | --- | --- |
| `sentinelinspect/inference_service/` | 5 | Stage 6 builds these |
| `sentinelinspect/inference/` core | `contracts.py`, `model_loader.py`, `batch_predict.py` | Stage 4 |
| `sentinelinspect/evaluation/` | `metrics.py`, `reports.py` | Stage 4, extracting from `evaluate.py` |
| `scripts/` | 5 wrappers + `start_api.sh` | Redundant: `python -m` already works. Delete. |
| `sentinelinspect/mlops/` | 4 | Out of scope per the plan. Delete. |
| `sentinelinspect/monitoring/` | 3 | Out of scope. Delete. |
| `sentinelinspect/jobs/` | 3 | Out of scope. Delete. |
| `sentinelinspect/utils/` | 4 | Speculative. Delete until something needs them. |
| `sentinelinspect/data/dataset.py`, `schemas.py` | 2 | Superseded: the dataset lives in `datamodule.py`. Delete. |
| `sentinelinspect/preprocessing/preprocess.py` | 1 | Superseded by `transforms.py`. Delete. |
| `sentinelinspect/training/callbacks.py`, `reproducibility.py` | 2 | Superseded: Lightning provides both. Delete. |
| `sentinelinspect/models/factory.py` | — | **Filled in Stage 1.** |
| `pyproject.toml`, `.github/workflows/ci.yaml`, `docker/` | 4 | Stages 2 and 8 |

Roughly 24 of the 39 should simply be deleted. An empty file is a promise the
repository does not keep.

---

### E · One fact, many homes

The ImageNet normalisation constants appear in **eight** places:

```
sentinelinspect/preprocessing/transforms.py   DEFAULTS
sentinelinspect/data/datamodule.py            fallback dict (now redundant)
configs/train.yaml                preprocessing block
configs/eval.yaml                 preprocessing block
configs/inference.yaml            preprocessing block
sentinelinspect/explainability/utils.py       IMAGENET_MEAN / IMAGENET_STD
sentinelinspect/explainability/shap_utils.py  inline, twice
```

The three config files are legitimate — that is configuration. The code copies
are not: change the normalisation and explainability silently disagrees with
training. `transforms.DEFAULTS` should be the only default in code.
