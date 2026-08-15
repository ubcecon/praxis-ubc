"""Reads the labels the model produced when we ran it.

Every model run in this notebook was done once on a GPU and saved to `cache/`, keyed
by the model, the prompt and the comment. The notebook reads those answers back, so
it runs anywhere in seconds and gives the same numbers every time.
"""

import hashlib
import json
from pathlib import Path

from . import config as cfg


class MissingAnswers(KeyError):
    pass


def cache_path(prompt: str, adapter: str | None = None) -> Path:
    slug = cfg.MODEL_ID.replace("/", "__") + (f"__{adapter}" if adapter else "")
    return cfg.CACHE_DIR / f"{slug}__{hashlib.sha256(prompt.encode()).hexdigest()[:16]}.json"


def cached_records(prompt: str, adapter: str | None = None) -> dict:
    """The saved run for one prompt: {comment_id: {label, reason, raw}}."""
    path = cache_path(prompt, adapter)
    if not path.exists():
        raise MissingAnswers(
            f"No saved answers at {path.name}. The prompt text is part of the key, "
            f"so editing the prompt means there is nothing to read back."
        )
    return json.loads(path.read_text(encoding="utf-8"))


def annotate(comment_ids, prompt, adapter=None):
    """Return "yes"/"no" for each comment id, from the saved run of this prompt."""
    saved = cached_records(prompt, adapter)
    ids = [str(c) for c in comment_ids]
    missing = [c for c in ids if c not in saved]
    if missing:
        raise MissingAnswers(
            f"{len(missing)} of {len(ids)} comments are not in "
            f"{cache_path(prompt, adapter).name}."
        )
    return [saved[c]["label"] for c in ids]
