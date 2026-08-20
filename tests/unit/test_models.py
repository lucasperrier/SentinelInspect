import pytest
import torch
import torch.nn as nn

from src.models.base import STAGES, CrackClassifier
from src.models.factory import MODEL_REGISTRY, build_model, model_class_for
from src.models.resnet50 import ResNet50Module
from src.models.vit import VisionTransformerModule


class TinyClassifier(CrackClassifier):
    """A stand-in backbone so the shared behaviour can be tested in milliseconds.

    Being able to write this at all is the point of the base class: everything
    under test here is inherited, not backbone-specific.
    """

    def build_backbone(self) -> nn.Module:
        return nn.Sequential(nn.Flatten(), nn.Linear(3 * 4 * 4, 2))


def _tiny(**overrides):
    cfg = {"name": "tiny", "model": "tiny", "num_classes": 2,
           "learning_rate": 1e-3, "weight_decay": 1e-4, "optimizer": "adam", "scheduler": None}
    cfg.update(overrides)
    return TinyClassifier(cfg)


def _batch(n=4):
    return torch.randn(n, 3, 4, 4), torch.randint(0, 2, (n,))


# --------------------------------------------------------------------------
# the factory
# --------------------------------------------------------------------------

def test_factory_dispatches_on_exact_name():
    assert model_class_for("resnet50") is ResNet50Module
    assert model_class_for("vit") is VisionTransformerModule


def test_factory_rejects_an_unknown_name():
    """The old dispatch was `if "vit" in name else ResNet50Module`, so a typo
    silently produced a ResNet. Now it fails and names what is registered."""
    with pytest.raises(ValueError, match="resnet50"):
        model_class_for("resnett50")
    with pytest.raises(ValueError, match="Unknown model name"):
        build_model({"name": "vision_transformer"})


def test_every_registered_model_can_be_built():
    built = {
        "resnet50": build_model({"name": "resnet50", "model": "resnet18", "pretrained": False}),
        "vit": build_model({"name": "vit", "model": "vit_tiny_patch16_224", "pretrained": False}),
    }
    assert set(built) == set(MODEL_REGISTRY)
    for name, model in built.items():
        assert isinstance(model, MODEL_REGISTRY[name])


# --------------------------------------------------------------------------
# shared behaviour
# --------------------------------------------------------------------------

def test_forward_returns_one_logit_per_class():
    x, _ = _batch(4)
    assert _tiny()(x).shape == (4, 2)


def test_shared_step_runs_for_every_stage():
    """`resnet50.py` used if/elif/elif with no else, so an unexpected stage
    raised UnboundLocalError from an unbound variable rather than saying so."""
    model = _tiny()
    for stage in STAGES:
        loss = model._shared_step(_batch(), stage)
        assert torch.isfinite(loss)


def test_unknown_stage_fails_with_a_useful_message():
    with pytest.raises(ValueError, match="Unknown stage"):
        _tiny()._shared_step(_batch(), "validation")


def test_each_stage_owns_its_metric_instances():
    """Torchmetrics accumulate state. Sharing one object between train and val
    would blend the two into a single meaningless number."""
    model = _tiny()
    seen = set()
    for stage in STAGES:
        for key in ("acc", "f1", "auc"):
            metric = model.stage_metrics(stage)[key]
            assert id(metric) not in seen
            seen.add(id(metric))
    assert len(seen) == 9


def test_metric_state_does_not_leak_between_stages():
    model = _tiny()
    model._shared_step(_batch(8), "train")
    assert model.stage_metrics("val")["acc"].update_count == 0


# --------------------------------------------------------------------------
# optimisers
# --------------------------------------------------------------------------

@pytest.mark.parametrize("name,expected", [
    ("adam", torch.optim.Adam),
    ("adamw", torch.optim.AdamW),
    ("sgd", torch.optim.SGD),
])
def test_configure_optimizers_supports_each_optimizer(name, expected):
    assert isinstance(_tiny(optimizer=name).configure_optimizers(), expected)


def test_unsupported_optimizer_is_rejected():
    """Previously an unknown name silently fell through to Adam."""
    with pytest.raises(ValueError, match="Unsupported optimizer"):
        _tiny(optimizer="lion").configure_optimizers()


def test_unsupported_scheduler_is_rejected():
    with pytest.raises(ValueError, match="Unsupported scheduler"):
        _tiny(scheduler="onecycle").configure_optimizers()


def test_cosine_scheduler_is_returned_with_the_optimizer():
    out = _tiny(scheduler="cosine", epochs=5).configure_optimizers()
    assert isinstance(out, dict)
    assert isinstance(out["lr_scheduler"], torch.optim.lr_scheduler.CosineAnnealingLR)


def test_subclass_defaults_apply_when_config_omits_them():
    """ResNet defaults to Adam at 1e-3, ViT to AdamW at 1e-4. The base class
    reads those from the subclass, not from a hardcoded value."""
    resnet = build_model({"name": "resnet50", "model": "resnet18", "pretrained": False})
    vit = build_model({"name": "vit", "model": "vit_tiny_patch16_224", "pretrained": False})

    assert (resnet.learning_rate, resnet.optimizer_name) == (1e-3, "adam")
    assert (vit.learning_rate, vit.optimizer_name) == (1e-4, "adamw")
