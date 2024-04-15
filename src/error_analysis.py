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


def plot_matrix(preds, gts, weights, normalize=True, norm_axis=1):
    mat0 = calc_conditional_prob(
        gts, gts, weights, normalize=normalize, norm_axis=norm_axis
    )
    mat1 = calc_conditional_prob(
        preds, gts, weights, normalize=normalize, norm_axis=norm_axis
    )
    _, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(15, 4))
    heatmap(
        mat0,
        annot=True,
        fmt=".2f",
        cmap="jet",
        xticklabels=LABELS,
        yticklabels=LABELS,
        square=True,
        ax=ax0,
        vmin=0,
        vmax=1,
    )
    ax0.set(
        xlabel="Ground Truth", ylabel="Ground Truth", title="Conditional Prob (GT-GT)"
    )
    heatmap(
        mat1,
        annot=True,
        fmt=".2f",
        cmap="jet",
        xticklabels=LABELS,
        yticklabels=LABELS,
        square=True,
        ax=ax1,
        vmin=0,
        vmax=1,
    )
    ax1.set(
        xlabel="Predictions", ylabel="Ground Truth", title="Conditional Prob (GT-Pred)"
    )
    heatmap(
        np.abs(mat1 - mat0),
        annot=True,
        fmt=".2f",
        cmap="jet",
        xticklabels=LABELS,
        yticklabels=LABELS,
        square=True,
        ax=ax2,
    )
    ax2.set(
        xlabel="Predictions",
        ylabel="Ground Truth",
        title="Difference (GT-Pred) - (GT-GT)",
    )


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

    plot_matrix(preds, gts, weights, normalize=False)
    savefig_or_show(save_file, root_path / "joint_prob.jpg")
    plot_matrix(preds, gts, weights)
    savefig_or_show(save_file, root_path / "cond_norm_gt.jpg")
    plot_matrix(preds, gts, weights, norm_axis=0)
    savefig_or_show(save_file, root_path / "cond_norm_pred.jpg")


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
