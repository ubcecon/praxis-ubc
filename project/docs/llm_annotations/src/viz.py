"""Figures for the notebook.

Palette: slots 1 and 2 of the validated default (blue/orange). Worst-pair colour
separation is well clear of the colourblind threshold, and both clear 3:1 contrast
on the chart surface. Chrome is deliberately recessive.
"""

import numpy as np

from . import config as cfg

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

YES = "#2a78d6"     # constructive
NO = "#eb6834"      # not constructive


def style():
    import matplotlib as mpl
    mpl.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": AXIS, "axes.linewidth": 0.8,
        "axes.labelcolor": INK_2, "axes.titlecolor": INK,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
        "grid.linestyle": "-", "axes.axisbelow": True,
        "xtick.color": MUTED, "ytick.color": MUTED,
        "xtick.labelsize": 9, "ytick.labelsize": 9,
        "axes.titlesize": 11, "axes.labelsize": 10,
        "legend.frameon": False, "legend.fontsize": 9,
        "font.family": "sans-serif", "figure.dpi": 110,
    })


def _clean(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


# --------------------------------------------------------------------------- #
# what the model can be read off, and where it points
# --------------------------------------------------------------------------- #
def probe_accuracy(hidden, labels, folds=5, C=0.01):
    """How much of the label a straight line can recover from the model's state.

    Strong regularisation because there are 300 comments and 2048 numbers each.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score

    return np.array([
        cross_val_score(LogisticRegression(max_iter=3000, C=C),
                        X.astype(np.float32), labels, cv=folds).mean()
        for X in hidden
    ])


def learning_curve(npz, path, accuracy=None):
    """Left: agreement with the human labels on held-out comments, per checkpoint,
    Right: how separable the label is from the model's internal state.

    The probe comments were never trained on, so the left panel is a genuine
    held-out learning curve: it says when training stopped helping.
    """
    import matplotlib.pyplot as plt

    from .data import _draws, _kappa

    style()
    steps, p, y = npz["steps"], npz["p_yes"], npz["labels"]
    acc = probe_accuracy(npz["hidden"], y) if accuracy is None else accuracy

    preds = (p > 0.5).astype(int)
    kappa = np.array([float(_kappa(y, q)) for q in preds])
    draws = _draws(len(y))
    band = np.array([np.percentile(_kappa(y[draws], q[draws]), [2.5, 97.5]) for q in preds])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2),
                                   gridspec_kw={"width_ratios": [1.35, 1]})

    ax1.fill_between(steps, band[:, 0], band[:, 1], color=YES, alpha=0.16, linewidth=0)
    ax1.plot(steps, kappa, color=YES, linewidth=2.4, marker="o", markersize=5,
             markeredgecolor=SURFACE, markeredgewidth=0.8)
    ax1.set_ylim(0.25, 0.9)
    ax1.set_xlabel("training step")
    ax1.set_ylabel("agreement with humans (kappa)")
    ax1.set_title("Does more training help?", loc="left")
    _clean(ax1)

    ax2.plot(steps, acc, color=YES, linewidth=2.4, marker="o", markersize=5,
             markeredgecolor=SURFACE, markeredgewidth=0.8)
    ax2.set_ylim(0.4, 1.0)
    ax2.set_xlabel("training step")
    ax2.set_ylabel("probe accuracy")
    ax2.set_title("What the model already knew", loc="left")
    _clean(ax2)

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return {"steps": steps.tolist(), "kappa": kappa, "probe_accuracy": acc}


def comparison_grid(truth, before, after, path, names=("zero-shot", "fine-tuned"),
                    positive="yes"):
    """One square per comment, wrong ones highlighted, before and after.

    Comments are ordered so the ones humans called constructive sit above the line,
    which makes a lopsided error pattern visible as a block rather than a number.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch

    style()
    truth = np.asarray(truth)
    order = np.argsort(truth != positive, kind="stable")
    truth = truth[order]
    n = len(truth)
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))
    split = (truth == positive).sum() / cols - 0.5

    right = "#dbe7f6"
    cmap = ListedColormap([right, NO])

    fig, axes = plt.subplots(1, 2, figsize=(10.2, 5.0))
    counts = []
    for ax, (name, pred) in zip(axes, zip(names, (before, after))):
        wrong = (np.asarray(pred)[order] != truth).astype(int)
        counts.append(int(wrong.sum()))
        padded = np.full(rows * cols, np.nan)
        padded[:n] = wrong
        ax.imshow(padded.reshape(rows, cols), cmap=cmap, vmin=0, vmax=1,
                  interpolation="nearest")
        ax.axhline(split, color=INK, linewidth=1.2)
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        for side in ax.spines.values():
            side.set_visible(False)
        ax.set_title(f"{name}\n{wrong.sum()} of {n} comments wrong", loc="left")

    axes[0].text(-0.8, split / 2, "humans said\nconstructive", ha="right", va="center",
                 fontsize=9, color=INK_2)
    axes[0].text(-0.8, (split + rows) / 2, "humans said\nnot constructive",
                 ha="right", va="center", fontsize=9, color=INK_2)

    fig.legend(handles=[Patch(facecolor=right, label="model agreed with the humans"),
                        Patch(facecolor=NO, label="model got it wrong")],
               loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("Each square is one comment", x=0.125, ha="left", fontsize=12, color=INK)
    fig.tight_layout(rect=[0, 0.04, 1, 0.95])
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return counts
