"""Uncertainty for the numbers the notebook reports.

Every number in this notebook is a number about a sample: 400 comments, or 214
doubly-labelled ones, or the 81 that are in both sets. A different sample would have given
a different number. These functions say how different.

Two ideas run through the module.

**Resampling.** To ask how much a number would move, draw a new sample of the same size
from the one we have, with replacement, and recompute. Do that ten thousand times and the
middle 95% of the answers is a confidence interval. This is the percentile bootstrap.

**Pairing.** When two raters are scored on the *same* comments, they are not two
independent measurements. A comment that is hard for one is usually hard for the other, and
that shared difficulty cancels out of the comparison. Resampling the comments once per
replicate and scoring both raters on that same resample keeps the pairing, and gives a much
tighter interval than treating them as unrelated. `bootstrap_kappa_diff_ci` is paired,
because in this notebook the two things being compared always saw the same comments.

Everything is seeded off `cfg.SEED`, so two runs give identical numbers.
"""

import numpy as np

from . import config as cfg
from .data import _kappa

__all__ = ["as_binary", "bootstrap_kappa_ci", "bootstrap_kappa_diff_ci", "fmt_ci"]


def as_binary(labels, positive="yes") -> np.ndarray:
    """Turn "yes"/"no" (or anything already 0/1) into a 0/1 array."""
    arr = np.asarray(list(labels))
    if arr.dtype.kind in "iub":
        return arr.astype(int)
    if arr.dtype.kind == "f":
        return (arr > 0.5).astype(int)
    return (np.char.lower(arr.astype(str)) == positive).astype(int)


def _index_draws(n, n_boot, seed):
    """One row per replicate, each row n comment positions drawn with replacement."""
    return np.random.default_rng(seed).integers(0, n, (n_boot, n))


def _percentile_ci(draws, alpha):
    lo, hi = np.percentile(draws, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def bootstrap_kappa_ci(y_true, y_pred, n_boot=10_000, seed=cfg.SEED, alpha=0.05) -> dict:
    """Cohen's kappa with a percentile bootstrap confidence interval.

    Comments are resampled with replacement, so the interval answers "if we had
    hand-labelled a different 400 comments from the same pool, how much would this kappa
    move?" It does not cover the labels themselves being wrong.

    Returns {"kappa", "lo", "hi", "n"}.
    """
    h, m = as_binary(y_true), as_binary(y_pred)
    if len(h) != len(m):
        raise ValueError(f"{len(h)} true labels but {len(m)} predictions")
    draws = _index_draws(len(h), n_boot, seed)
    lo, hi = _percentile_ci(_kappa(h[draws], m[draws]), alpha)
    return {"kappa": float(_kappa(h, m)), "lo": lo, "hi": hi, "n": len(h)}


def bootstrap_kappa_diff_ci(y_true, pred_a, pred_b, n_boot=10_000, seed=cfg.SEED,
                            alpha=0.05) -> dict:
    """Confidence interval on kappa(B) - kappa(A), for two raters scored on the same items.

    The two are paired: each replicate draws one set of comment positions and scores both
    on it, so a resample that happens to be easy is easy for both and the difficulty
    cancels. Treating them as independent would widen this interval a lot and would be the
    wrong comparison.

    If the interval excludes zero, B beat A by more than resampling noise explains.

    Returns {"a", "b", "diff", "lo", "hi", "n"}.
    """
    h, a, b = as_binary(y_true), as_binary(pred_a), as_binary(pred_b)
    if not len(h) == len(a) == len(b):
        raise ValueError("all three label vectors must describe the same comments")
    draws = _index_draws(len(h), n_boot, seed)
    ht = h[draws]
    lo, hi = _percentile_ci(_kappa(ht, b[draws]) - _kappa(ht, a[draws]), alpha)
    ka, kb = float(_kappa(h, a)), float(_kappa(h, b))
    return {"a": ka, "b": kb, "diff": kb - ka, "lo": lo, "hi": hi, "n": len(h)}


def fmt_ci(lo, hi, places=2) -> str:
    """"[0.70, 0.83]", for putting an interval next to a number in prose."""
    return f"[{lo:.{places}f}, {hi:.{places}f}]"
