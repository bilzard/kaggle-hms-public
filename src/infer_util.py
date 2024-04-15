from pathlib import Path

import numpy as np
import polars as pl
from scipy.special import softmax

from src.constant import LABELS
from src.preprocess import process_label, select_develop_samples


def load_metadata(
    data_dir: Path,
    phase: str,
    fold_split_dir: Path,
    fold: int = -1,
    group_by_eeg: bool = False,
    weight_key: str = "weight_per_eeg",
    num_samples: int = 128,
    min_weight: float = 0.0,
) -> pl.DataFrame:
    match phase:
        case "train":
            fold_split_df = pl.read_parquet(fold_split_dir / "fold_split.pqt")
            if fold >= 0:
                fold_split_df = fold_split_df.filter(pl.col("fold").eq(fold))
            target_fold = fold_split_df.select("eeg_id", "fold").unique()
            metadata = pl.read_csv(data_dir / "train.csv")
            metadata = metadata.join(target_fold, on="eeg_id")

            metadata = process_label(metadata)

            # min_weightより大きいweightのみを使用
            metadata = metadata.filter(pl.col("weight").ge(min_weight))

            if group_by_eeg:
                metadata = metadata.group_by("eeg_id", maintain_order=True).agg(
                    pl.col("spectrogram_id").first(),
                    pl.col("label_id").first(),
                    pl.col(weight_key).first().alias("weight"),
                    *[pl.col(f"{label}_prob_per_eeg").first() for label in LABELS],
                )

            return metadata
        case "test":
            metadata = pl.read_csv(data_dir / "test.csv")
            metadata = process_label(metadata, add_dummy_label=True)

            return metadata
        case "develop":
            metadata = pl.read_csv(data_dir / "train.csv")
            metadata = process_label(metadata)
            return select_develop_samples(metadata, num_samples=num_samples)
        case _:
            raise ValueError(f"Invalid phase: {phase}")


def make_submission(
    eeg_ids: np.ndarray,
    predictions: np.ndarray,
    apply_softmax: bool = True,
    T: float = 1.0,
):
    if apply_softmax:
        predictions = softmax(predictions / T, axis=1)
    submission_df = pl.DataFrame(
        dict(eeg_id=eeg_ids)
        | {f"{label}_vote": predictions[:, i] for i, label in enumerate(LABELS)}
    )
    return submission_df
