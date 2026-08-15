"""The New York Times comment pairs, and the scores the models gave them.

Each pair is one comment an editor picked and one they did not, taken from the same
article and matched on length, so length cannot be used to tell them apart.

The scores were produced once on a GPU and saved. `load_scores` reads them back, keyed
the same way the labels are: model, adapter, comment.
"""

import numpy as np
import pandas as pd

from . import config as cfg

MODELS = {"zero-shot": "zero-shot",
          "fine-tuned": "naive_adapter",
          "length-balanced": "balanced_adapter"}


def load_pairs() -> pd.DataFrame:
    """One row per comment; two rows share a pair_id."""
    pairs = pd.read_csv(cfg.NYT_PAIRS_CSV)
    pairs["comment_id"] = pairs.comment_id.astype(str)
    return pairs


def load_scores(pairs=None) -> pd.DataFrame:
    """How sure each model was that a comment is an editor's pick, 0 to 1."""
    pairs = load_pairs() if pairs is None else pairs
    saved = np.load(cfg.NYT_SCORES, allow_pickle=True)
    order = {c: i for i, c in enumerate(saved["comment_id"].astype(str))}
    rows = [order[c] for c in pairs.comment_id]
    return pairs.assign(**{name: saved[key][rows] for name, key in MODELS.items()})


def side_by_side(scored, model) -> pd.DataFrame:
    """One row per pair: the pick's score and its partner's."""
    return scored.pivot(index="pair_id", columns="is_pick", values=model)


def game_score(scored, model) -> float:
    """Share of pairs where the model scored the editor's pick above its partner."""
    w = side_by_side(scored, model)
    return float(((w["yes"] > w["no"]).astype(float) + 0.5 * (w["yes"] == w["no"])).mean())


def labels(scored, model, bar=0.5) -> pd.Series:
    """Turn scores into yes/no answers at a given bar."""
    return pd.Series(np.where(scored[model] > bar, "yes", "no"), index=scored.index)


def pick_rate(scored, model, bar=0.5) -> float:
    """Share of all comments the model called an editor's pick."""
    return float((scored[model] > bar).mean())


def example_pair(scored, missed_by, separated_by, seed=cfg.SEED):
    """A real pair one model calls a pick twice and the other tells apart.

    Returns the two comment texts and both models' scores, for reading rather than
    for counting.
    """
    a, b = side_by_side(scored, missed_by), side_by_side(scored, separated_by)
    bar_a, bar_b = cfg.NYT_BAR[missed_by], cfg.NYT_BAR[separated_by]
    ok = a.index[(a["yes"] > bar_a) & (a["no"] > bar_a)
                 & (b["yes"] > bar_b) & (b["no"] <= bar_b)]
    text = scored.pivot(index="pair_id", columns="is_pick", values="comment_text")
    shortest = text.loc[ok].map(len).sum(axis=1).sort_values().index
    pair_id = shortest[int(np.random.default_rng(seed).integers(min(5, len(shortest))))]
    return {"pick": text.loc[pair_id, "yes"], "other": text.loc[pair_id, "no"],
            missed_by: (a.loc[pair_id, "yes"], a.loc[pair_id, "no"]),
            separated_by: (b.loc[pair_id, "yes"], b.loc[pair_id, "no"])}
