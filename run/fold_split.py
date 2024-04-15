from pathlib import Path

import hydra
import numpy as np
import polars as pl
from sklearn.model_selection import GroupKFold

from src.config import MainConfig
from src.preprocess import process_label
from src.proc_util import trace


def split_group_k_fold(metadata: pl.DataFrame, num_splits: int) -> pl.DataFrame:
    group_kfold = GroupKFold(n_splits=num_splits)
    df = metadata.select("eeg_id", "patient_id").unique(maintain_order=True).to_pandas()

    df["fold"] = -1
    for fold, (_, valid_idxs) in enumerate(
        group_kfold.split(df[["eeg_id"]], df["eeg_id"], df["patient_id"])
    ):
        df.loc[valid_idxs, "fold"] = fold
    df["fold"] = df["fold"].astype(np.int8)

    return pl.from_pandas(df)


@hydra.main(config_path="conf", config_name="main", version_base="1.2")
def main(cfg: MainConfig):
    data_dir = Path(cfg.env.data_dir)
    metadata = pl.read_csv(data_dir / f"{cfg.phase}.csv")
    metadata = process_label(metadata)

    with trace("split fold"):
        if cfg.split.strategy == "group_k_fold":
            df = split_group_k_fold(metadata, cfg.split.num_splits)
        else:
            raise ValueError(f"Unknown split strategy: {cfg.split.strategy}")

        print(df.group_by("fold", maintain_order=True).agg(pl.col("eeg_id").count()))
        print("-" * 40)
        print(df.head())

    if not cfg.dry_run:
        with trace("save fold split"):
            df.write_parquet("fold_split.pqt")

            for fold, this_df in df.group_by("fold", maintain_order=True):
                eeg_ids = this_df["eeg_id"].to_numpy()
                np.save(f"fold_{fold}.npy", eeg_ids)

            eeg_ids = df["eeg_id"].to_numpy()
            np.save("fold_all.npy", eeg_ids)


if __name__ == "__main__":
    main()
