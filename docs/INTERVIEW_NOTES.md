# Interview notes

Not documentation. This is the defence: what the project is, why each decision
went the way it did, and what is honestly wrong with it.

---

## The thirty-second version

> SentinelInspect is a crack-detection classifier turned into a small deployable
> system. The interesting part is not the model — it is that the dataset is a
> versioned contract with leakage validation, that training, offline evaluation
> and the HTTP API all run through one inference core so they cannot disagree,
> and that uncertain predictions are routed to human review instead of being
> forced into a decision.

If they only ask one follow-up, it will be about the leakage bug. Lead with it.

---

## The numbers, and what they mean

```
test accuracy 0.9885 · f1 0.9886 · auc 0.9990 · recall(crack) 0.9899   (5,889 images)
triage: 6.5% routed to review, catching 79.4% of errors
        auto-decided accuracy 0.9975 on the remaining 5,506
```

Frozen-backbone ResNet-50, 3 epochs, CPU. Say the caveats yourself: the dataset is easy,
the backbone is frozen because there was no GPU, and the confidence is uncalibrated.

**If asked why it dropped from 99.78%:** that figure came from a contaminated test set.
This one is real. Two independent code paths agree on it — Lightning's own `trainer.test`
and the `Predictor` used by the API both return 0.9885, which is incidental proof that
offline evaluation and serving share one forward pass.

**The triage number is the interesting one.** Routing 6.5% of images to a human intercepts
79.4% of the model's mistakes and lifts accuracy on everything decided automatically to
99.75%. The band was tuned on validation for 80% error recall and hit 79.4% on test, so it
generalised. That is the argument for tuning under a review-capacity budget rather than
picking round numbers: the guessed [0.35, 0.65] caught only 58%.

---

## Architecture in one breath

```
data/raw/**  ─build_manifest→  manifest.csv  ─splitters→  train/val/test.csv
                                    │                            │
                                    └────── validate_dataset ────┘
                                                 │
                                          CrackDataModule
                                                 │
                        ┌────────────────────────┼────────────────────────┐
                     train.py                Predictor                 (same
                  Hydra + MLflow      load once · preprocess · forward   core)
                        │             softmax · triage · contract
                        │                        │
                   checkpoint ───────────────────┼──────────────┐
                                                 │              │
                                        evaluate.py        FastAPI /predict
                                        CLI predict        FastAPI /health
```

The single most important line: **every prediction goes through
`Predictor._probabilities`.** Three adapters over one core cannot drift; three
copies of the same twenty lines always do.

---

## The stories

### 1. The leakage bug — lead with this

`data/raw/sdnet2018/` was a byte-identical copy of `data/raw/ccic/`: same
SHA256 set, same filenames. Two archives of the same Kaggle/Mendeley dataset,
extracted twice, one misnamed.

Splits were assigned by hashing `relative_path`. `ccic/Positive/00001.jpg` and
`sdnet2018/Positive/00001.jpg` are different strings, so the same image scored
differently and landed in different splits.

```
CONTENT overlap (SHA256)      PATH overlap (what the validator checked)
  train ↔ val:   8,426          train ↔ val:   0
  train ↔ test:  8,482          train ↔ test:  0   ← reported "PASSED"
  val   ↔ test:  1,969          val   ↔ test:  0
```

**8,482 test images had been trained on, and the validator passed.** The
previously reported 99.78% accuracy was meaningless.

The fix: key the split on `sha256` so identical bytes always co-locate, and run
the overlap check over content as well as paths.

> **The line to say:** *"My own validator passed while 70% of my test set was
> contaminated, because I was validating identity instead of content. The
> manifest had computed the hash all along — nothing consumed it."*

**Follow-up they will ask: how do you know it is fixed?**
Two tests. One asserts every `sha256` maps to exactly one split. The other pins
the *old* behaviour — that path-keying genuinely does separate duplicates —
because otherwise the first test could pass vacuously on a fixture with no
duplicates.

### 2. A config block that had never done anything

`configs/train.yaml` carried a full `preprocessing:` section. `train.py` passed
a flat dict; `transforms.py` called `cfg.get("preprocessing", {})`, expecting a
nested one, found nothing, and silently used hardcoded defaults.

Every augmentation setting in that file had never affected a run. `.get()` with
a default never raises — that is why it survived for months.

The unit tests were green the whole time, because they called the function with
a correctly nested dict. **They tested the contract; nobody upheld it.** The fix
was to change the signature so the function takes the block itself, reject
unknown keys, and add an integration test that starts from the real YAML.

### 3. Explainability had been broken for five months

`run_explainability.py` constructed the datamodule with `val_split`,
`test_split`, `robustness_split` — parameters deleted in an earlier refactor.
836 lines that raised `TypeError` on the first call.

Worse: `docs/roadmap.md` warned about exactly this stale signature but named
`train.py` and `evaluate.py`, which had already been fixed. **A stale doc
pointing at the wrong file is worse than no doc** — it makes you check, find
nothing, and conclude everything is fine.

Underneath were three more: `scikit-image` was an undeclared dependency so SHAP
had never once run (the failure was swallowed into a `[WARN]` line);
`vit_grid_from_name` ignored its argument and returned a hardcoded `(14, 14)`,
correct only for patch-16; and the KernelSHAP unpacking assumed a pre-0.45 array
shape that NumPy 2 no longer coerces to a scalar.

### 4. Renaming a directory broke the dataset

The manifest stored an absolute `path` column. Renaming the project produced
`missing files: 40000`.

> **The line to say:** *"A data contract that breaks when you rename a folder
> isn't portable. The manifest now stores a path relative to a configured root
> plus a content hash — nothing machine-specific."*

The proof it worked: after regenerating, the splits came out **byte-identical**
(28,054 / 6,057 / 5,889). The split is a pure function of content, so changing
the manifest schema moved nothing.

### 5. A loss that was quietly wrong

Evaluation averaged per-batch mean losses. With `batch_size=64` and 5,889 test
samples, the final batch of 25 counted as much as a full one.

The test pins it with numbers: batches of 64 at loss 0.1 and 1 at loss 10.0 give
5.05 the naive way and 0.252 weighted by sample count.

---

## Design decisions, and the trade-off each one accepts

**Why hash the filename instead of `train_test_split(random_state=42)`?**
Because a hash is a pure function of the item. Add 5,000 images tomorrow and
every existing image keeps its split; a seeded shuffle reshuffles everything and
silently invalidates comparisons with yesterday's run. *Trade-off:* class balance
becomes approximate rather than exact — measured drift is under one point.
Exact stratification would require ranking within each class, which makes an
item's split depend on every other item and destroys the stability property.
**Those two properties are mutually exclusive, and stability is worth more.**

**Why is the review band two-sided?**
0.48 and 0.52 are equally uncertain. A one-sided confidence floor flags only one
of them. *Trade-off:* the band is a hyperparameter, so it is tuned on validation
under a review-rate budget — unconstrained, "catch more errors" always returns
[0, 1].

**Why does offline evaluation go through the Predictor?**
So a reported metric and a served decision come from identical arithmetic. The
original code had a torchvision transform in `predict.py` and albumentations
everywhere else. I measured the skew: max tensor delta 0.035. Small — but
uncontrolled, and it would have grown.

**Why does the checkpoint override the config's architecture?**
The artifact knows what it is. Trusting `configs/model/*.yaml` instead means
editing a YAML silently invalidates weights on disk; you find out through a wall
of shape-mismatch errors at deploy time. A warning fires on mismatch.

**Why a frozen backbone?**
No GPU. A full fine-tune of ResNet-50 on 28k images is days on this CPU; a
linear probe is about 90 minutes. *Trade-off:* accuracy is lower than a full
fine-tune would reach. It is honest and reproducible, which matters more here
than the last few points.

**Why delete 26 files?**
They were empty. `src/mlops/registry.py` at 0 bytes reads as abandoned work, not
planned work. Anything not on the active plan should be absent, not hollow.

**Why no batch endpoint, model registry, or drift monitoring?**
Scope was one week, fixed in advance, with an explicit cut list. Shipping six
things that work beats twelve that half-work. I can describe how I would add
each; none of them is on the CV.

---

## Honest limitations — say these before you are asked

1. **CCIC is an easy dataset.** 227×227 centred crops, perfectly balanced,
   near-saturated. Real inspection imagery is not. The system design is the
   contribution; the task is not hard.
2. **Frozen backbone, no GPU.** A full fine-tune would score higher.
3. **One dataset, no cross-dataset generalisation.** The intended second source
   turned out to be a duplicate of the first. Real SDNET2018 (8% positive,
   three substrates) would be the honest stress test and is not done.
4. **The review band is tuned on one validation split**, not cross-validated.
5. **No authentication, rate limiting or persistence on the API.** Deliberate —
   it is a demonstration of the inference boundary, not a production service.
6. **1,598 duplicate images remain within CCIC itself.** Reported as a warning,
   not an error: duplication inside one split over-weights an image but does not
   invalidate a measurement. Cross-split duplication does, and that is an error.

---

## Questions to expect

**"Is this your work or the model's?"**
Answer plainly: the original pipeline is yours; you used Claude heavily to find
defects, restructure and add coverage; you can walk through any line. The commit
history says so via `Co-Authored-By` trailers. A vague answer is the only wrong
one.

**"Why is accuracy lower than what you had before?"**
Because the earlier number was measured on a contaminated test set. This one is
real. That answer is strictly better than a higher number you cannot defend.

**"What would you do next?"**
Real SDNET2018 for cross-dataset generalisation; calibration (the confidence is
a softmax output, not a calibrated probability — temperature scaling on
validation would fix that); and prediction logging so the review rate can be
tracked over time rather than measured once.

**"What is the weakest part?"**
The model. It is a linear probe on ImageNet features for a nearly-solved task.
Everything interesting is the engineering around it — which is exactly what the
project was for.
