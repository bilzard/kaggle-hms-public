import shutil
from pathlib import Path

import hydra
import numpy as np
import polars as pl
from sklearn.model_selection import GroupKFold

from src.config import MainConfig
from src.proc_util import trace


@hydra.main(config_path="conf", config_name="main", version_base="1.2")
def main(cfg: MainConfig):
    data_dir = Path(cfg.env.data_dir)
    output_dir = Path(cfg.env.output_dir)
    metadata = pl.read_csv(data_dir / f"{cfg.phase}.csv")

    if (not cfg.dry_run) and (cfg.cleanup) and (output_dir.exists()):
        shutil.rmtree(output_dir)
        print(f"Removed {cfg.phase} dir: {output_dir}")

    with trace("split fold"):
        group_kfold = GroupKFold(n_splits=cfg.split.num_splits)
        df = (
            metadata.select("eeg_id", "patient_id")
            .unique(maintain_order=True)
            .to_pandas()
        )

        df["fold"] = -1
        for fold, (_, valid_idxs) in enumerate(
            group_kfold.split(df[["eeg_id"]], df["eeg_id"], df["patient_id"])
        ):
            df.loc[valid_idxs, "fold"] = fold
        df["fold"] = df["fold"].astype(np.int8)

        if not output_dir.exists():
            output_dir.mkdir(parents=True)

        print(df.groupby("fold").agg({"eeg_id": "count"}))
        print("-" * 40)
        print(df.head())

    if not cfg.dry_run:
        with trace("save fold split"):
            df.to_parquet(output_dir / "fold_split.pqt", index=False)

            for fold, this_df in df.groupby("fold"):
                eeg_ids = this_df["eeg_id"].to_numpy()
                np.save(output_dir / f"fold_{fold}.npy", eeg_ids)

            eeg_ids = df["eeg_id"].to_numpy()
            np.save(output_dir / "fold_all.npy", eeg_ids)


if __name__ == "__main__":
    main()
