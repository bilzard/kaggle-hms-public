from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import kl_div, softmax
from seaborn import heatmap

from src.constant import LABELS


def calc_conditional_prob(
    preds: np.ndarray,
    gts: np.ndarray,
    weights: np.ndarray | None = None,
    normalize: bool = True,
    norm_axis: int = 0,
    eps=1e-4,
):
    """
    Parameters
    ----------
    preds: (N, C) array of predicted probabilities
    gts: (N, C) array of ground truth probabilities
    weights: (N, ) array of weights

    Returns
    -------
    conditional_matrix: (C, C) array of conditional probabilities
    """
    gts = gts[:, :, np.newaxis]
    preds = preds[:, np.newaxis, :]
    if weights is not None:
        weights = weights[:, np.newaxis, np.newaxis]
        mat = (preds * gts * weights).sum(axis=0) / weights.sum(axis=0)
    else:
        mat = (preds * gts).mean(axis=0)

    if normalize:
        mat = mat / (mat.sum(axis=norm_axis, keepdims=True) + eps)
    return mat


def savefig_or_show(save_file: bool, file_name: Path) -> None:
    if save_file:
        plt.savefig(file_name, bbox_inches="tight")
    else:
        plt.tight_layout()
        plt.show()


def plot_confusion_matrix(
    preds: np.ndarray,
    gts: np.ndarray,
    weights: np.ndarray,
    save_file: bool = False,
    apply_softmax: bool = True,
    root_path: Path = Path("figure"),
) -> None:
    """
    Parameters
    ----------
    preds: (N, C) array of predicted probabilities
    gts: (N, C) array of ground truth probabilities
    weights: (N, ) array of weights
    """
    root_path = Path(root_path)
    if apply_softmax:
        preds = softmax(preds, axis=1)

    _, ax = plt.subplots()
    heatmap(
        calc_conditional_prob(preds, gts, weights, normalize=False),
        cmap="viridis",
        xticklabels=LABELS,
        yticklabels=LABELS,
        ax=ax,
        annot=True,
        fmt=".2f",
        square=True,
    )
    ax.set(xlabel="Predicted", ylabel="Ground Truth", title="P(GT=i, pred=j)")
    savefig_or_show(save_file, root_path / "total_prob.jpg")

    _, ax = plt.subplots()
    heatmap(
        calc_conditional_prob(preds, gts, weights, norm_axis=1),
        cmap="viridis",
        xticklabels=LABELS,
        yticklabels=LABELS,
        ax=ax,
        annot=True,
        fmt=".2f",
        square=True,
    )
    ax.set(xlabel="Predicted", ylabel="Ground Truth", title="P(pred=j | GT=i)")
    savefig_or_show(save_file, root_path / "conditional_by_gt.jpg")

    _, ax = plt.subplots()
    heatmap(
        calc_conditional_prob(preds, gts, weights),
        cmap="viridis",
        xticklabels=LABELS,
        yticklabels=LABELS,
        ax=ax,
        annot=True,
        fmt=".2f",
        square=True,
    )
    ax.set(xlabel="Predicted", ylabel="Ground Truth", title="P(GT=i | pred=j)")
    savefig_or_show(save_file, root_path / "conditional_by_pred.jpg")


def plot_loss_distribution(
    preds: np.ndarray,
    gts: np.ndarray,
    weights: np.ndarray,
    eps: float = 1e-4,
    save_file: bool = False,
    root_path: Path = Path("figure"),
    apply_softmax: bool = True,
):
    if apply_softmax:
        preds = softmax(preds, axis=1)
    kl_divs = kl_div(gts, preds).sum(axis=1)
    log_kl_divs = np.log(kl_divs + eps)

    _, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].hist(kl_divs, bins=50, alpha=0.5)
    axes[0].set(xlabel="KL Divergence", ylabel="count")
    axes[1].hist(log_kl_divs, bins=100, alpha=0.5)
    axes[1].set(xlabel="log(KL Divergence)", ylabel="count")

    savefig_or_show(save_file, root_path / "kl_div_distribution.jpg")
