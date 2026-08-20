"""Content hashing.

A module of its own, with no third-party imports, because both the data layer
and the inference layer need it. Leaving it in `build_manifest` meant importing
pandas just to fingerprint a checkpoint -- and through that, dragging the whole
data stack into the serving image.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

CHUNK_SIZE = 1024 * 1024


def sha256_file(path: str | Path, chunk_size: int = CHUNK_SIZE) -> str:
    """Streaming SHA256 of a file's contents.

    Read in chunks so a multi-hundred-megabyte checkpoint never lands in memory
    all at once.
    """
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()
