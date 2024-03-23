from pathlib import Path

import hydra
import numpy as np
import polars as pl
from scipy.special import kl_div, softmax

from src.config import EnsembleMainConfig
from src.constant import LABELS
from src.infer_util import load_metadata, make_submission
from src.proc_util import trace


def calc_metric(gts, preds):
    losses = kl_div(gts, softmax(preds, axis=1))
    loss_per_label = losses.sum(axis=1)
    loss = loss_per_label.mean()
    return loss


def simple_evaluate(metadata: pl.DataFrame, predictions: pl.DataFrame) -> float:
    eval_df = metadata.join(predictions, on="eeg_id")
    gts = eval_df.select(f"{label}_prob_per_eeg" for label in LABELS).to_numpy()
    preds = eval_df.select(f"{label}_vote" for label in LABELS).to_numpy()

    loss = calc_metric(gts, preds)
    return loss


def do_evaluate_with_weight(
    metadata: pl.DataFrame,
    predictions: pl.DataFrame,
):
    eval_df = metadata.join(predictions, on="eeg_id")

    gts = eval_df.select(f"{label}_prob_per_eeg" for label in LABELS).to_numpy()
    preds = eval_df.select(f"{label}_vote" for label in LABELS).to_numpy()
    weights = eval_df["weight"].to_numpy()
    eeg_ids = eval_df["eeg_id"].to_list()

    losses = kl_div(gts, softmax(preds, axis=1))
    loss_per_label = (losses * weights[:, np.newaxis]).sum(axis=0) / weights.sum()

    loss = loss_per_label.sum()
    loss_per_label = dict(zip(LABELS, loss_per_label.tolist()))

    losses = losses.sum(axis=1)
    loss_df = pl.DataFrame(dict(eeg_id=eeg_ids, loss=losses))

    return loss, loss_per_label, loss_df


def do_evaluate(
    metadata: pl.DataFrame,
    predictions: pl.DataFrame,
    apply_label_weight: bool = True,
):
    if apply_label_weight:
        loss, loss_per_label, loss_df = do_evaluate_with_weight(metadata, predictions)
        print(f"* loss: {loss:.4f}")
        print(
            "* loss_per_label:",
            ", ".join([f"{k}={v:.4f}" for k, v in loss_per_label.items()]),
        )
        loss_df.write_parquet("losses.pqt")
    else:
        loss = simple_evaluate(metadata, predictions)
        print(f"* loss: {loss:.4f}")


@hydra.main(config_path="conf", config_name="ensemble", version_base="1.2")
def main(cfg: EnsembleMainConfig):
    working_dir = Path(cfg.env.working_dir)
    fold_split_dir = Path(working_dir / "fold_split" / cfg.phase)
    infer_dir = Path(working_dir / "infer")

    ensemble_entity = cfg.ensemble_entity

    with trace("load predictions"):
        print(f"ensemble_name: {ensemble_entity.name}")
        num_seeds = 0
        predictions = []
        for experiment in ensemble_entity.experiments:
            local_preds = []
            for ensemble_fold in experiment.folds:
                fold = ensemble_fold.split
                for seed in ensemble_fold.seeds:
                    local_preds.append(
                        pl.read_parquet(
                            infer_dir
                            / experiment.exp_name
                            / f"fold_{fold}"
                            / f"seed_{seed:d}"
                            / f"pred_{cfg.phase}.pqt"
                        )
                    )
                    num_seeds += 1
            local_preds = pl.concat(local_preds)
            local_preds = local_preds.group_by("eeg_id", maintain_order=True).agg(
                pl.col(f"{label}_vote").mean() for label in LABELS
            )
            predictions.append(local_preds)

        print(
            f"* ensemble of {len(ensemble_entity.experiments)} models (total seeds={num_seeds})"
        )

        predictions = pl.concat(predictions)
        predictions = predictions.group_by("eeg_id", maintain_order=True).agg(
            pl.col(f"{label}_vote").mean() for label in LABELS
        )

        match cfg.phase:
            case "train":
                metadata = load_metadata(
                    data_dir=Path(cfg.env.data_dir),
                    phase=cfg.phase,
                    fold_split_dir=fold_split_dir,
                    group_by_eeg=True,
                    weight_key="weight_per_eeg",
                    num_samples=cfg.dev.num_samples,
                )
                do_evaluate(
                    metadata, predictions, apply_label_weight=cfg.apply_label_weight
                )

            case "test":
                submission_df = make_submission(
                    eeg_ids=predictions["eeg_id"].to_list(),
                    predictions=predictions.drop("eeg_id").to_numpy(),
                    apply_softmax=True,
                )
                submission_dir = Path(cfg.env.submission_dir)
                submission_df.write_csv(submission_dir / "submission.csv")
                print(submission_df)


if __name__ == "__main__":
    main()
