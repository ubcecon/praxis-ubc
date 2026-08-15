"""Paths and constants. Everything the notebook reads is saved in this folder."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

SEED = 20260810

DATA = ROOT / "data"
ARTIFACTS = DATA / "artifacts"
CACHE_DIR = ROOT / "cache"
MEDIA = ROOT / "media"

GOLD_CSV = DATA / "socc_gold_labels.csv"
C3_CSV = DATA / "c3_labelled_comments.csv"
ARTICLES_CSV = DATA / "globe_articles.csv"
CMV_CSV = DATA / "cmv_eval_set.csv"

MODEL_ID = "unsloth/Qwen3-1.7B-unsloth-bnb-4bit"
ADAPTER = "qlora-notebook"      # names the fine-tuned run in the saved answers

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
