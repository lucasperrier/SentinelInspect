"""SentinelInspect -- a visual inspection triage system.

The package is organised as a pipeline, each layer consuming only the previous
layer's output:

    data          manifest, deterministic splits, integrity validation
    preprocessing the one transform pipeline training and serving share
    models        Lightning modules over timm backbones, plus a registry
    training      the Hydra + MLflow training entrypoint
    evaluation    offline scoring into a reports bundle
    inference     single-image prediction
"""

__version__ = "0.1.0"
