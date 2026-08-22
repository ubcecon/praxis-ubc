#Contains supporter commands to run the GPU specific model inference

import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = ROOT / "cache"

MODEL_ID = "unsloth/Qwen3-1.7B-unsloth-bnb-4bit"
ADAPTER = "qlora-notebook"
ADAPTER_BALANCED = "qlora-balanced"

SEED = 20260810
N_EVAL_PER_CLASS = 200          # balanced held-out evaluation set
N_PROBE_PER_CLASS = 150         # disjoint set used to look inside the model

RUBRIC = """You are annotating reader comments from a Canadian news website.

A comment is CONSTRUCTIVE if it tries to add something to the conversation: it \
makes a specific point, gives evidence or a personal experience, offers a \
solution, or engages with the article's argument.

A comment is NOT CONSTRUCTIVE if it is only an insult, a one-line dismissal, \
sarcasm with no substance, off-topic ranting, or an unsupported assertion.

Comment:
\"\"\"{comment}\"\"\"

Reply in exactly this format and nothing else:
LABEL: yes
REASON: <one short sentence>"""


# the split
def balanced_sample(df, label_col, n_per_class, seed, positive="yes", exclude=None):
    """n_per_class rows of each class, shuffled."""
    pool = df.drop(index=exclude) if exclude is not None else df
    rng = np.random.default_rng(seed)
    parts = []
    for value in (positive, *[v for v in pool[label_col].unique() if v != positive]):
        sub = pool[pool[label_col] == value]
        parts.append(sub.iloc[rng.choice(len(sub), n_per_class, replace=False)])
    return pd.concat(parts).sample(frac=1, random_state=seed)


def gold_splits(gold):
    """The evaluation set and a disjoint probe set, both balanced."""
    evaluation = balanced_sample(gold, "is_constructive", N_EVAL_PER_CLASS, SEED)
    probe = balanced_sample(gold, "is_constructive", N_PROBE_PER_CLASS,
                            SEED + 1, exclude=evaluation.index)
    assert not set(evaluation.index) & set(probe.index)
    return evaluation.reset_index(drop=True), probe.reset_index(drop=True)


# the saved answers
class MissingAnswers(KeyError):
    pass


def cache_path(prompt, adapter=None):
    slug = MODEL_ID.replace("/", "__") + (f"__{adapter}" if adapter else "")
    return CACHE_DIR / f"{slug}__{hashlib.sha256(prompt.encode()).hexdigest()[:16]}.json"


def cached_records(prompt, adapter=None):
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
