"""Download both corpora and write the csv files in data/. Needs the network, not a GPU.

    python model_call_scripts/build_data.py            # both
    python model_call_scripts/build_data.py --civil
    python model_call_scripts/build_data.py --detox

Civil Comments supplies the training, evaluation and probe sets. Wikipedia Detox supplies the transfer set. Run this before gpu_run.py, which scores whatever this writes.
"""

import argparse
import json
import sys
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from model_call_scripts import paths  # noqa: E402  (sets HF_HOME on import)
from model_call_scripts.cached_run import (  # noqa: E402
    DATA, N_EVAL_PER_CLASS, N_PROBE_PER_CLASS, N_TRAIN, SEED, TOXIC_BAR, balanced_sample,
)

# ------------------------------------------------------------------ civil comments

# The plain HuggingFace build drops article_id, and the split needs it.
REPO = "pietrolesci/civilcomments-wilds"
SOURCE_FILE = "raw/train-00000-of-00001.parquet"

KEEP = ["comment_id", "comment_text", "toxicity", "is_toxic", "n_raters", "n_toxic_votes",
        "article_id", "publication_id", "created_date", "rating", "n_chars", "n_words"]

IDENTITY = ["male", "female", "transgender", "other_gender", "heterosexual",
            "homosexual_gay_or_lesbian", "bisexual", "other_sexual_orientation",
            "christian", "jewish", "muslim", "hindu", "buddhist", "atheist",
            "other_religion", "black", "white", "asian", "latino",
            "other_race_or_ethnicity", "physical_disability",
            "intellectual_or_learning_disability", "psychiatric_or_mental_illness",
            "other_disability"]


def load_civil():
    from huggingface_hub import hf_hub_download
    frame = pd.read_parquet(hf_hub_download(REPO, SOURCE_FILE, repo_type="dataset"))
    print(f"source: {len(frame):,} comments, {frame.article_id.nunique():,} articles")

    frame = frame.rename(columns={"id": "comment_id"})
    frame["comment_id"] = frame.comment_id.astype(str)
    frame["comment_text"] = frame.comment_text.astype(str).str.strip()
    frame = frame[frame.comment_text.str.len() > 0]

    frame["is_toxic"] = np.where(frame.toxicity >= TOXIC_BAR, "yes", "no")
    frame["n_raters"] = frame.toxicity_annotator_count.astype(int)

    # toxicity is a share of the panel, so share times panel size gives the vote count. It lands on a whole number every time, which the assert checks.
    votes = frame.toxicity * frame.n_raters
    assert np.abs(votes - votes.round()).max() < 1e-6, "vote counts are not whole numbers"
    frame["n_toxic_votes"] = votes.round().astype(int)

    frame["n_chars"] = frame.comment_text.str.len()
    frame["n_words"] = frame.comment_text.str.split().str.len()
    frame["mentions_identity"] = frame[IDENTITY].max(axis=1) >= 0.5
    return frame


def split_by_article(frame, rng):
    """Deal whole articles into three pools so no comment thread spans two of them."""
    articles = frame.article_id.unique()
    articles = articles[rng.permutation(len(articles))]
    cuts = (int(0.15 * len(articles)), int(0.30 * len(articles)))
    pools = {
        "eval": frame[frame.article_id.isin(articles[: cuts[0]])],
        "probe": frame[frame.article_id.isin(articles[cuts[0]: cuts[1]])],
        "train": frame[frame.article_id.isin(articles[cuts[1]:])],
    }
    for name, pool in pools.items():
        yes = int((pool.is_toxic == "yes").sum())
        print(f"  {name:>5}: {len(pool):>7,} comments, {pool.article_id.nunique():>6,} "
              f"articles, {yes:>6,} toxic ({yes / len(pool):.1%})")
    return pools


def build_civil():
    frame = load_civil()
    toxic = frame.is_toxic == "yes"
    print(f"  toxic at {TOXIC_BAR}: {toxic.sum():,} ({toxic.mean():.2%})")
    print(f"  panel size: median {frame.n_raters.median():.0f}, "
          f"min {frame.n_raters.min()}, max {frame.n_raters.max()}")

    print("\nsplitting by article")
    rng = np.random.default_rng(SEED)
    pools = split_by_article(frame, rng)

    evaluation = balanced_sample(pools["eval"], "is_toxic", N_EVAL_PER_CLASS, SEED)
    probe = balanced_sample(pools["probe"], "is_toxic", N_PROBE_PER_CLASS, SEED + 1)

    # Half and half, not the natural 11%. At the natural rate a thousand comments carry about a hundred toxic ones, too few for the adapter to learn that class.
    train = balanced_sample(pools["train"], "is_toxic", N_TRAIN // 2, SEED + 2)

    cols = KEEP + ["mentions_identity"] + IDENTITY
    for name, out in (("civil_eval", evaluation), ("civil_probe", probe),
                      ("civil_train", train)):
        # utf-8-sig: Excel reads a csv without the byte-order mark as cp1252 and turns every curly quote into mojibake. Readers have to match it.
        out[cols].to_csv(DATA / f"{name}.csv", index=False, encoding="utf-8-sig")
        yes = int((out.is_toxic == "yes").sum())
        print(f"\n[ok] {name}.csv  {len(out)} comments, {yes} toxic, "
              f"{out.article_id.nunique()} articles")

    scored = set(evaluation.article_id) | set(probe.article_id)
    assert not set(evaluation.article_id) & set(probe.article_id), "eval and probe overlap"
    assert not set(train.article_id) & scored, "an article is both trained on and scored"
    print("\nno article is both trained on and scored")

    frame[["comment_id", "n_raters", "n_toxic_votes", "toxicity"]].to_csv(
        DATA / "rater_votes.csv", index=False, encoding="utf-8-sig")
    print(f"[ok] rater_votes.csv  {len(frame):,} vote splits, for the human ceiling")


# ----------------------------------------------------------------- wikipedia detox

FIGSHARE = "https://api.figshare.com/v2/articles/4563973/files"
DETOX_CACHE = paths.DETOX       # 114 MB of source tsv, kept out of the repo
N_PER_CLASS = 400


def fetch(name):
    DETOX_CACHE.mkdir(parents=True, exist_ok=True)
    local = DETOX_CACHE / name
    if local.exists():
        print(f"  {name}: already downloaded")
        return local
    files = {f["name"]: f for f in json.load(urllib.request.urlopen(FIGSHARE, timeout=60))}
    if name not in files:
        raise SystemExit(f"{name} is not in figshare item 4563973: {sorted(files)}")
    print(f"  {name}: downloading {files[name]['size'] / 1e6:.0f} MB")
    req = urllib.request.Request(files[name]["download_url"],
                                 headers={"User-Agent": "Mozilla/5.0"})
    local.write_bytes(urllib.request.urlopen(req, timeout=900).read())
    return local


def clean(text):
    """Detox comments come from revision diffs, so whitespace and quotes arrive escaped."""
    return (str(text).replace("NEWLINE_TOKEN", "\n")
                     .replace("TAB_TOKEN", "\t")
                     .replace("`", '"')
                     .strip())


def build_detox():
    print("wikipedia detox, figshare item 4563973")
    comments = pd.read_csv(fetch("toxicity_annotated_comments.tsv"), sep="\t")
    votes = pd.read_csv(fetch("toxicity_annotations.tsv"), sep="\t")
    print(f"\nsource: {len(comments):,} comments, {len(votes):,} rater judgements")
    print(f"  sample: {comments['sample'].value_counts().to_dict()}")

    # The blocked half was drawn from around user-block events, so its toxic rate is a property of the sampling and not of Wikipedia.
    frame = comments[comments["sample"] == "random"].copy()
    print(f"  keeping the random half: {len(frame):,}")

    agg = votes.groupby("rev_id").toxicity.agg(["mean", "size"])
    agg.columns = ["share_toxic", "n_raters"]
    frame = frame.join(agg, on="rev_id").dropna(subset=["share_toxic", "comment"])
    frame["n_raters"] = frame.n_raters.astype(int)
    frame["n_toxic_votes"] = (frame.share_toxic * frame.n_raters).round().astype(int)
    frame["is_toxic"] = np.where(frame.share_toxic >= TOXIC_BAR, "yes", "no")

    toxic = frame.is_toxic == "yes"
    print(f"  toxic at {TOXIC_BAR}: {toxic.sum():,} of {len(frame):,} ({toxic.mean():.3%})")
    print(f"  panel size: median {frame.n_raters.median():.0f}, "
          f"min {frame.n_raters.min()}, max {frame.n_raters.max()}")
    print(f"  years {int(frame.year.min())} to {int(frame.year.max())}")

    before = frame.comment.iloc[0]
    frame["comment_text"] = frame.comment.map(clean)
    frame = frame[frame.comment_text.str.len() > 0]
    print(f"\n  before cleanup: {before[:110]}")
    print(f"  after:          {frame.comment_text.iloc[0][:110]!r}")

    frame["comment_id"] = "detox_" + frame.rev_id.astype("int64").astype(str)
    frame["n_chars"] = frame.comment_text.str.len()
    frame["n_words"] = frame.comment_text.str.split().str.len()

    # Balanced to match the Civil Comments evaluation set. Kappa moves with the base rate, so comparing two sets at different rates compares the rates as much as the models.
    rng = np.random.default_rng(SEED)
    parts = []
    for label in ("yes", "no"):
        pool = frame[frame.is_toxic == label]
        if len(pool) < N_PER_CLASS:
            raise SystemExit(f"only {len(pool)} {label} comments, wanted {N_PER_CLASS}")
        parts.append(pool.iloc[rng.choice(len(pool), N_PER_CLASS, replace=False)])
    sample = pd.concat(parts).sample(frac=1, random_state=SEED).reset_index(drop=True)

    cols = ["comment_id", "comment_text", "is_toxic", "share_toxic", "n_raters",
            "n_toxic_votes", "n_chars", "n_words", "year", "ns", "logged_in"]
    sample[cols].to_csv(DATA / "detox_eval.csv", index=False, encoding="utf-8-sig")
    frame[["comment_id", "n_raters", "n_toxic_votes", "share_toxic"]].to_csv(
        DATA / "detox_votes.csv", index=False, encoding="utf-8-sig")

    print(f"\n[ok] detox_eval.csv   {len(sample)} comments, "
          f"{(sample.is_toxic == 'yes').sum()} toxic")
    print(f"[ok] detox_votes.csv  {len(frame):,} vote splits")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--civil", action="store_true")
    ap.add_argument("--detox", action="store_true")
    args = ap.parse_args()
    both = not (args.civil or args.detox)
    if args.civil or both:
        build_civil()
    if args.detox or both:
        if both:
            print()
        build_detox()
    return 0


if __name__ == "__main__":
    sys.exit(main())
