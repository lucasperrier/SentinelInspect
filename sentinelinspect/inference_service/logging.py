"""Structured request logging.

One JSON line per request. Plain text is fine to read and useless to query;
JSON means latency and review rate can be aggregated later without parsing
prose.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from typing import Any, Dict, Optional

LOGGER_NAME = "sentinelinspect.service"


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: Dict[str, Any] = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "event": record.getMessage(),
        }
        extra = getattr(record, "context", None)
        if isinstance(extra, dict):
            payload.update(extra)
        if record.exc_info:
            payload["error"] = self.formatException(record.exc_info)
        return json.dumps(payload)


def configure_logging(level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(level)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(JsonFormatter())
        logger.addHandler(handler)
        logger.propagate = False
    return logger


def log_event(event: str, level: int = logging.INFO, **context: Any) -> None:
    logging.getLogger(LOGGER_NAME).log(level, event, extra={"context": context})


class Timer:
    """Wall-clock timing for a block, in milliseconds."""

    def __enter__(self) -> "Timer":
        self._start = time.perf_counter()
        self.elapsed_ms = 0.0
        return self

    def __exit__(self, *exc: Any) -> None:
        self.elapsed_ms = (time.perf_counter() - self._start) * 1000.0
