"""Loading, splitting and scoring the hand-labelled comments."""

import numpy as np
import pandas as pd

from . import config as cfg


def load_gold() -> pd.DataFrame:
    """The 1,043 comments a human labelled for constructiveness."""
    return pd.read_csv(cfg.GOLD_CSV)


def balanced_sample(df, label_col, n_per_class, seed, positive="yes", exclude=None):
    """n_per_class rows of each class, shuffled."""
    pool = df.drop(index=exclude) if exclude is not None else df
    rng = np.random.default_rng(seed)
    parts = []
    for value in (positive, *[v for v in pool[label_col].unique() if v != positive]):
        sub = pool[pool[label_col] == value]
        parts.append(sub.iloc[rng.choice(len(sub), n_per_class, replace=False)])
    return pd.concat(parts).sample(frac=1, random_state=seed)


def gold_splits(gold=None):
    """The evaluation set and a disjoint probe set, both balanced."""
    gold = load_gold() if gold is None else gold
    evaluation = balanced_sample(gold, "is_constructive", cfg.N_EVAL_PER_CLASS, cfg.SEED)
    probe = balanced_sample(gold, "is_constructive", cfg.N_PROBE_PER_CLASS,
                            cfg.SEED + 1, exclude=evaluation.index)
    assert not set(evaluation.index) & set(probe.index)
    return evaluation.reset_index(drop=True), probe.reset_index(drop=True)


def length_by_class(gold=None) -> pd.DataFrame:
    """How long is a comment of each class? The clue the model can learn instead."""
    gold = load_gold() if gold is None else gold
    rows = []
    for label, name in (("yes", "constructive"), ("no", "not constructive")):
        n = gold.loc[gold.is_constructive == label, "comment_text"].str.len()
        rows.append({"humans said": name, "comments": len(n),
                     "median length": f"{n.median():.0f} characters",
                     "longest tenth": f"over {n.quantile(0.9):.0f} characters"})
    return pd.DataFrame(rows).set_index("humans said")


# ------------------------------------------------------------------ scoring
def _binary(human, model):
    human, model = list(human), list(model)
    ok = [i for i, p in enumerate(model) if p in ("yes", "no")]
    return (np.array([1 if human[i] == "yes" else 0 for i in ok]),
            np.array([1 if model[i] == "yes" else 0 for i in ok]),
            len(model) - len(ok))


def _kappa(h, m):
    """Cohen's kappa, vectorised over a leading draw axis so resampling is cheap."""
    po = (h == m).mean(axis=-1)
    ph, pm = h.mean(axis=-1), m.mean(axis=-1)
    pe = ph * pm + (1 - ph) * (1 - pm)
    return np.where(pe < 1, (po - pe) / np.where(pe < 1, 1 - pe, 1), 0.0)


def _draws(n, n_draws=4000):
    return np.random.default_rng(cfg.SEED).integers(0, n, (n_draws, n))


def metrics(human, model) -> dict:
    """Agreement between two label vectors of "yes"/"no"."""
    from sklearn.metrics import accuracy_score, f1_score

    h, m, unparsed = _binary(human, model)
    return {"accuracy": accuracy_score(h, m),
            "f1": f1_score(h, m, zero_division=0),
            "cohen_kappa": float(_kappa(h, m)),
            "predicted_yes": float(np.mean(m)),
            "unparsed": unparsed}


SCOREBOARD = {"accuracy": "accuracy",
              "cohen_kappa": "agreement (kappa)",
              "predicted_yes": "calls it constructive"}


def scoreboard(human, models: dict) -> pd.DataFrame:
    """The three numbers this notebook argues from, one column per model.

    `metrics` returns more than these three. The rest are there when you want them, but a
    table of numbers nobody explains is worse than a smaller table, so this is what gets
    shown: how often the model is right, how much of that it earned, and how often it says
    "constructive" at all.
    """
    return pd.DataFrame({
        name: {label: metrics(human, answers)[key] for key, label in SCOREBOARD.items()}
        for name, answers in models.items()
    })


def confusion(human, model, names=("constructive", "not constructive")) -> pd.DataFrame:
    """Counts with both axes spelled out, so neither has to be decoded."""
    yes, no = names
    t = pd.crosstab(pd.Series(list(human)), pd.Series(list(model)))
    t = t.reindex(index=["yes", "no"], columns=["yes", "no"], fill_value=0)
    t.index = [f"humans said {yes}", f"humans said {no}"]
    t.columns = [f"model said {yes}", f"model said {no}"]
    t.index.name = t.columns.name = None
    t["share the model agreed"] = (np.diag(t.to_numpy()) / t.sum(axis=1)).round(2)
    return t
