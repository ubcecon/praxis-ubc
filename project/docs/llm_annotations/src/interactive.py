"""Which comments changed when the model was fine-tuned.

Section 6 counts how many comments each model got wrong. These two functions say *which*
ones, so the comments fine-tuning fixed and the ones it broke can be read rather than only
counted. A model that improves on average still gets specific things wrong that it used to
get right, and the only way to judge that trade is to look at what was traded.
"""

import numpy as np
import pandas as pd


def _shorten(text, n=200):
    """One line of a comment, short enough to sit in a table cell."""
    text = " ".join(str(text).split())
    return text if len(text) <= n else text[: n - 1] + "…"


def transition_frame(truth, zero_shot, fine_tuned, texts) -> pd.DataFrame:
    """One row per evaluation comment, with its label under each of the three raters.

    Adds `zs_ok` and `ft_ok`, which record whether each model matched the human label.
    Together those two columns place every comment in one of four groups: right both
    times, wrong both times, fixed by fine-tuning, or broken by it.
    """
    frame = pd.DataFrame({
        "truth": np.asarray(list(truth)),
        "zero_shot": np.asarray(list(zero_shot)),
        "fine_tuned": np.asarray(list(fine_tuned)),
        "text": np.asarray(list(texts)),
    })
    frame["zs_ok"] = np.where(frame.zero_shot == frame.truth, "right", "wrong")
    frame["ft_ok"] = np.where(frame.fine_tuned == frame.truth, "right", "wrong")
    return frame


def flow_table(frame, before="wrong", after="right", chars=200) -> pd.DataFrame:
    """Every comment in one of those four groups, for reading rather than counting.

    `before="wrong", after="right"` is the set of comments fine-tuning fixed;
    `before="right", after="wrong"` is the set it broke.
    """
    rows = frame[(frame.zs_ok == before) & (frame.ft_ok == after)]
    return pd.DataFrame({
        "humans said": np.where(rows.truth == "yes", "constructive", "not constructive"),
        "zero-shot said": np.where(rows.zero_shot == "yes", "constructive", "not constructive"),
        "fine-tuned said": np.where(rows.fine_tuned == "yes", "constructive", "not constructive"),
        "comment": [_shorten(t, chars) for t in rows.text],
    }).reset_index(drop=True)
