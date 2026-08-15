"""Loading, splitting and scoring."""

import hashlib
import re

import numpy as np
import pandas as pd

from . import config as cfg


def load_gold() -> pd.DataFrame:
    return pd.read_csv(cfg.GOLD_CSV)


def load_c3() -> pd.DataFrame:
    return pd.read_csv(cfg.C3_CSV)


def _norm_hash(text) -> str:
    return hashlib.sha1(re.sub(r"\s+", " ", str(text)).strip().lower().encode()).hexdigest()


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


def c3_training_split(c3=None, gold=None, n_per_class=1000):
    """The comments the model was fine-tuned on, with any gold comments removed first."""
    c3 = load_c3() if c3 is None else c3
    gold = load_gold() if gold is None else gold
    leaked = set(gold.comment_text.map(_norm_hash))
    clean = c3.loc[~c3.comment_text.map(_norm_hash).isin(leaked)]
    clean = clean.dropna(subset=["comment_text", "constructive_binary"])
    clean = clean.assign(label=np.where(clean.constructive_binary == 1, "yes", "no"))
    # Matches how the training set was drawn; see the fine-tuning section.
    train = balanced_sample(clean, "label", 100, cfg.SEED)
    train = balanced_sample(clean, "label", 250, cfg.SEED + 2, exclude=train.index)
    return clean, balanced_sample(clean, "label", n_per_class, cfg.SEED + 3,
                                  exclude=train.index).reset_index(drop=True)


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
    out = {"accuracy": accuracy_score(h, m),
           "f1": f1_score(h, m, zero_division=0),
           "cohen_kappa": float(_kappa(h, m))}
    out["predicted_yes"] = float(np.mean(m))
    out["unparsed"] = unparsed
    return out


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


def length_report(cmv, gold) -> pd.DataFrame:
    """Are the two classes the same length? The confound worth checking."""
    rows = []
    for name, series in (
        ("CMV persuasive", cmv.loc[cmv.persuasive == "yes", "comment_text"]),
        ("CMV not persuasive", cmv.loc[cmv.persuasive == "no", "comment_text"]),
        ("SOCC constructive", gold.loc[gold.is_constructive == "yes", "comment_text"]),
        ("SOCC not constructive", gold.loc[gold.is_constructive == "no", "comment_text"]),
    ):
        n = series.str.len()
        rows.append({"set": name, "n": len(n), "median chars": int(n.median()),
                     "p90 chars": int(n.quantile(0.9))})
    return pd.DataFrame(rows)
