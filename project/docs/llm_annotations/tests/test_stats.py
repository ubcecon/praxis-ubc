"""Check src/stats.py against an independent implementation.

Run with `python -m pytest tests -q` from the notebook folder.

The kappa itself is compared against scikit-learn rather than a hand-copied expected value.
The bootstrap intervals have no independent implementation to check against, so the tests
check the properties that must hold instead: determinism, correct ordering, and that
pairing does what pairing is supposed to do.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config as cfg  # noqa: E402
from src import stats  # noqa: E402
from src.data import _kappa  # noqa: E402


@pytest.fixture
def toy():
    """400 labels with a deliberately lopsided predictor, like the zero-shot model."""
    rng = np.random.default_rng(0)
    truth = rng.integers(0, 2, 400)
    pred = np.where(rng.random(400) < 0.75, truth, 1 - truth)
    worse = np.where(rng.random(400) < 0.55, truth, 1 - truth)
    return truth, pred, worse


def test_kappa_matches_sklearn(toy):
    from sklearn.metrics import cohen_kappa_score

    truth, pred, _ = toy
    assert float(_kappa(truth, pred)) == pytest.approx(cohen_kappa_score(truth, pred))


def test_bootstrap_kappa_point_estimate_matches_sklearn(toy):
    from sklearn.metrics import cohen_kappa_score

    truth, pred, _ = toy
    out = stats.bootstrap_kappa_ci(truth, pred, n_boot=500)
    assert out["kappa"] == pytest.approx(cohen_kappa_score(truth, pred))
    assert out["lo"] < out["kappa"] < out["hi"]
    assert out["n"] == 400


def test_bootstrap_is_deterministic(toy):
    truth, pred, _ = toy
    a = stats.bootstrap_kappa_ci(truth, pred, n_boot=500)
    b = stats.bootstrap_kappa_ci(truth, pred, n_boot=500)
    assert a == b


def test_kappa_accepts_yes_no_strings(toy):
    truth, pred, _ = toy
    words = lambda v: ["yes" if x else "no" for x in v]  # noqa: E731
    assert (stats.bootstrap_kappa_ci(words(truth), words(pred), n_boot=200)
            == stats.bootstrap_kappa_ci(truth, pred, n_boot=200))


def test_diff_ci_orders_the_two_raters(toy):
    truth, better, worse = toy
    out = stats.bootstrap_kappa_diff_ci(truth, worse, better, n_boot=2000)
    assert out["b"] > out["a"]
    assert out["diff"] == pytest.approx(out["b"] - out["a"])
    assert out["lo"] > 0, "a clearly better rater should have a CI above zero"


def test_diff_ci_of_a_rater_against_itself_straddles_zero(toy):
    truth, pred, _ = toy
    out = stats.bootstrap_kappa_diff_ci(truth, pred, pred, n_boot=500)
    assert out["diff"] == pytest.approx(0.0)
    assert out["lo"] == pytest.approx(0.0) and out["hi"] == pytest.approx(0.0)


def test_pairing_beats_treating_the_raters_as_independent(toy):
    """The paired interval must be narrower, which is the whole reason for pairing."""
    truth, better, worse = toy
    paired = stats.bootstrap_kappa_diff_ci(truth, worse, better, n_boot=4000)

    n = len(truth)
    rng = np.random.default_rng(cfg.SEED)
    d1, d2 = rng.integers(0, n, (4000, n)), rng.integers(0, n, (4000, n))
    unpaired = _kappa(truth[d2], better[d2]) - _kappa(truth[d1], worse[d1])
    lo, hi = np.percentile(unpaired, [2.5, 97.5])

    assert (paired["hi"] - paired["lo"]) < (hi - lo)


def test_mismatched_lengths_raise():
    with pytest.raises(ValueError):
        stats.bootstrap_kappa_ci([1, 0, 1], [1, 0])
    with pytest.raises(ValueError):
        stats.bootstrap_kappa_diff_ci([1, 0, 1], [1, 0, 1], [1, 0])


def test_as_binary_handles_the_shapes_the_notebook_passes():
    assert list(stats.as_binary(["yes", "no", "YES"])) == [1, 0, 1]
    assert list(stats.as_binary([1, 0, 1])) == [1, 0, 1]
    assert list(stats.as_binary([0.9, 0.1])) == [1, 0]


def test_fmt_ci():
    assert stats.fmt_ci(0.7003, 0.8251) == "[0.70, 0.83]"
