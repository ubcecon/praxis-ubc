"""Paths and constants. Everything the notebook reads is saved in this folder.

Paths come from environment variables so nothing here depends on one machine's disk
layout; the defaults are relative to this file and work as shipped.
"""

import os
from pathlib import Path

ROOT = Path(os.environ.get("LLM_ANNOT_ROOT", Path(__file__).resolve().parents[1]))

SEED = 20260810

DATA = ROOT / "data"
ARTIFACTS = DATA / "artifacts"
CACHE_DIR = Path(os.environ.get("LLM_ANNOT_CACHE", ROOT / "cache"))
MEDIA = ROOT / "media"

GOLD_CSV = DATA / "socc_gold_labels.csv"
C3_CSV = DATA / "c3_labelled_comments.csv"
ARTICLES_CSV = DATA / "globe_articles.csv"
NYT_PAIRS_CSV = DATA / "nyt_pairs.csv"
NYT_SCORES = ARTIFACTS / "nyt_p_yes.npz"

MODEL_ID = "unsloth/Qwen3-1.7B-unsloth-bnb-4bit"

# Two fine-tuned adapters. The first was trained on C3 as it comes; the second on a
# sample rebuilt so that the two classes have the same spread of lengths.
ADAPTER = "qlora-notebook"
ADAPTER_BALANCED = "qlora-balanced"

# Where each model's bar for saying "yes" was moved to, measured once on a separate set
# of 100 NYT pairs that is scored nowhere else in this notebook. Full precision on
# purpose: rounding these moves comments across the bar and changes the scores.
NYT_BAR = {
    "zero-shot": 0.964469850063324,
    "fine-tuned": 0.9951679110527039,
    "length-balanced": 0.7826507091522217,
}

MAX_SEQ = 512
TRAIN_STEPS = 200
N_EVAL_PER_CLASS = 200          # balanced held-out evaluation set
N_PROBE_PER_CLASS = 150         # disjoint set used to look inside the model


def summary() -> dict:
    return {
        "model": MODEL_ID,
        "seed": SEED,
        "eval set": f"{2 * N_EVAL_PER_CLASS} comments ({N_EVAL_PER_CLASS} per class)",
        "probe set": f"{2 * N_PROBE_PER_CLASS} comments",
        "saved answers": f"{len(list(CACHE_DIR.glob('*.json')))} model runs in cache/",
    }
