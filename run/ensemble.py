from pathlib import Path

import hydra
import numpy as np
import polars as pl
import torch
from torch.nn.functional import kl_div, log_softmax

from src.config import EnsembleMainConfig
from src.constant import LABELS
from src.infer_util import load_metadata, make_submission
from src.proc_util import trace


@hydra.main(config_path="conf", config_name="ensemble", version_base="1.2")
def main(cfg: EnsembleMainConfig):
    working_dir = Path(cfg.env.working_dir)
    fold_split_dir = Path(working_dir / "fold_split" / cfg.phase)
    infer_dir = Path(working_dir / "infer")

    ensemble_entity = cfg.ensemble_entity

    with trace("load predictions"):
        print(f"ensemble_name: {ensemble_entity.name}")
        prediction_paths: list[Path] = []
        for experiment in ensemble_entity.experiments:
            for ensemble_fold in experiment.folds:
                fold = ensemble_fold.split
                for seed in ensemble_fold.seeds:
                    prediction_paths.append(
                        infer_dir
                        / experiment.exp_name
                        / f"fold_{fold}"
                        / f"seed_{seed:d}"
                        / f"pred_{cfg.phase}.pqt"
                    )

        predictions = []
        for path in prediction_paths:
            assert path.exists()
            print(path)
            predictions.append(pl.read_parquet(path))

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
                )
                eval_df = metadata.join(predictions, on="eeg_id")

                gts = eval_df.select(f"{label}_prob" for label in LABELS).to_numpy()
                preds = eval_df.select(f"{label}_vote" for label in LABELS).to_numpy()
                weights = eval_df["weight"].to_numpy()

                gts = torch.from_numpy(gts.copy())
                preds = torch.from_numpy(preds.copy())
                weights = torch.from_numpy(weights.copy())

                with torch.no_grad():
                    loss_per_label = (
                        (
                            (
                                kl_div(
                                    log_softmax(preds, dim=1),
                                    gts,
                                    reduction="none",
                                )
                                * weights[:, np.newaxis]
                            ).sum(dim=0)
                            / weights.sum()
                        )
                        .detach()
                        .cpu()
                        .numpy()
                    )
                loss = loss_per_label.sum()
                loss_per_label = dict(zip(LABELS, loss_per_label.tolist()))
                print("loss:", loss)
                print(
                    "loss_per_label:",
                    ", ".join([f"{k}={v:.4f}" for k, v in loss_per_label.items()]),
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
