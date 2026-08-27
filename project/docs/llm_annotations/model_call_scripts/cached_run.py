"""Constants and cache access shared by the notebook and the GPU scripts.

The notebook imports this and nothing else. Everything here runs on CPU with no network.

RUBRIC lives here because its text is part of the cache filename. Change one character and `annotate` stops finding the saved answers, which is deliberate: answers written under a different prompt should not be read back as if they matched.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = ROOT / "cache"
DATA = ROOT / "data"

MODEL_ID = "unsloth/Qwen3-1.7B-unsloth-bnb-4bit"
ADAPTER = "qlora-toxicity"

SEED = 20260810

# The Civil Comments label is the share of a panel who called a comment toxic, so a comment rated by four people arrives as 0, 0.25, 0.5, 0.75 or 1. This turns it into a yes or no.
TOXIC_BAR = 0.5

N_EVAL_PER_CLASS = 100
N_PROBE_PER_CLASS = 100
N_TRAIN = 1000

RUBRIC = """You are rating reader comments from a news website for toxicity.

The people who labelled these comments were asked: how toxic is this comment? They \
answered on a four-point scale.

  Very Toxic     a very hateful, aggressive or disrespectful comment, or one very likely \
to make someone leave the discussion.
  Toxic          a rude, disrespectful or unreasonable comment, or one somewhat likely to \
make someone leave the discussion.
  Hard to Say    you cannot tell.
  Not Toxic      none of the above.

Answer yes if the comment is Toxic or Very Toxic. Answer no otherwise.

Comment:
\"\"\"{comment}\"\"\"

Reply in exactly this format and nothing else:
LABEL: yes
REASON: <one short sentence>"""


def balanced_sample(df, label_col, n_per_class, seed, positive="yes", exclude=None):
    """Equal numbers of each class, shuffled. Used by build_data.py."""
    pool = df.drop(index=exclude) if exclude is not None else df
    rng = np.random.default_rng(seed)
    parts = []
    for value in (positive, *[v for v in pool[label_col].unique() if v != positive]):
        sub = pool[pool[label_col] == value]
        parts.append(sub.iloc[rng.choice(len(sub), n_per_class, replace=False)])
    return pd.concat(parts).sample(frac=1, random_state=seed)


def read_set(name):
    """One csv from data/. comment_id stays a string so it never loses leading zeros."""
    return pd.read_csv(DATA / name, encoding="utf-8-sig", dtype={"comment_id": str})


class MissingAnswers(KeyError):
    pass


def cache_path(prompt, adapter=None):
    slug = MODEL_ID.replace("/", "__") + (f"__{adapter}" if adapter else "")
    return CACHE_DIR / f"{slug}__{hashlib.sha256(prompt.encode()).hexdigest()[:16]}.json"


def cached_records(prompt, adapter=None):
    """Everything saved for one model and prompt: {comment_id: {label, reason, raw}}."""
    path = cache_path(prompt, adapter)
    if not path.exists():
        raise MissingAnswers(
            f"No saved answers at {path.name}. The prompt is part of the filename, so "
            f"editing RUBRIC leaves nothing to read back."
        )
    return json.loads(path.read_text(encoding="utf-8"))


def annotate(comment_ids, prompt, adapter=None):
    """The saved yes/no for each comment. Raises on a miss; never calls a model."""
    saved = cached_records(prompt, adapter)
    ids = [str(c) for c in comment_ids]
    missing = [c for c in ids if c not in saved]
    if missing:
        raise MissingAnswers(f"{len(missing)} of {len(ids)} comments are not in "
                             f"{cache_path(prompt, adapter).name}.")
    return [saved[c]["label"] for c in ids]
