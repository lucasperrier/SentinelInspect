"""FastAPI application.

The model is loaded in the lifespan handler, before the first request is
accepted. A service that loads lazily reports healthy while it is still
incapable of answering, and the first caller pays the load latency.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from sentinelinspect import __version__
from sentinelinspect.inference_service.dependencies import build_predictor
from sentinelinspect.inference_service.logging import configure_logging, log_event
from sentinelinspect.inference_service.routes import router


def create_app(checkpoint_path: str | None = None, config_dir: str | None = None) -> FastAPI:
    configure_logging()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        try:
            app.state.predictor = build_predictor(checkpoint_path, config_dir)
            meta = app.state.predictor.metadata
            log_event(
                "startup.model_loaded",
                model=meta.name,
                backbone=meta.backbone,
                checkpoint_sha256=meta.checkpoint_sha256,
            )
        except Exception as exc:
            # Fail loudly at startup rather than 500ing on every request. The
            # container then exits and the orchestrator sees a crash loop,
            # which is the correct signal for a misconfigured deployment.
            log_event("startup.model_load_failed", level=logging.ERROR, error=str(exc))
            raise
        yield
        app.state.predictor = None
        log_event("shutdown.complete")

    app = FastAPI(
        title="SentinelInspect",
        version=__version__,
        summary="Visual inspection triage: classify an image and route uncertain cases to review.",
        lifespan=lifespan,
    )
    app.include_router(router)

    @app.get("/", include_in_schema=False)
    def root() -> JSONResponse:
        return JSONResponse({"service": "sentinelinspect", "version": __version__, "docs": "/docs"})

    return app


app = create_app()
