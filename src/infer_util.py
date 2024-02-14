from pathlib import Path

import numpy as np
import polars as pl
from scipy.special import softmax

from src.constant import LABELS
from src.preprocess import process_label


def load_metadata(
    data_dir: Path, phase: str, fold_split_dir: Path, fold: int = -1
) -> pl.DataFrame:
    """
    phaseに応じてmetadataをロードする
    train:
        - validationのeegのみをロード
        - eeg_idは重複あり
    test:
        - 全てのeegをロード
        - eeg_idは重複なし
    """
    match phase:
        case "train":
            fold_split_df = pl.read_parquet(fold_split_dir / "fold_split.pqt")
            eeg_ids = (
                fold_split_df.filter(pl.col("fold").eq(fold)).select("eeg_id").unique()
            )
            metadata = pl.read_csv(data_dir / "train.csv")
            metadata = metadata.join(eeg_ids, on="eeg_id")
            metadata = process_label(metadata)

            return metadata
        case "test":
            metadata = pl.read_csv(data_dir / "test.csv")
            metadata = process_label(metadata, is_test=True)
            return metadata
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
