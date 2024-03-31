# %%
from pathlib import Path
from shutil import rmtree

import numpy as np
import polars as pl

data_dir = Path("/ml-docker/input/hms-harmful-brain-activity-classification")
work_dir = Path("/ml-docker/working/kaggle-hms-bilzard")
out_dir = work_dir / "data/fold_split/train"

if out_dir.exists():
    rmtree(out_dir)

if not out_dir.exists():
    out_dir.mkdir(parents=True)

metadata = pl.read_csv(data_dir / "train.csv")
index_df = metadata.group_by("eeg_id").agg(pl.col("patient_id").first()).sort("eeg_id")

fold_df = pl.read_csv(work_dir / "data/fold_split_ari/patient_fold.csv")
fold_df = fold_df.select("patient_id", "fold").unique()
fold_df = index_df.join(fold_df, on="patient_id")

for fold in range(5):
    fold_eeg_ids = (
        fold_df.filter(pl.col("fold") == fold).get_column("eeg_id").to_numpy()
    )
    np.save(out_dir / f"fold_{fold}.npy", fold_eeg_ids)

fold_df.write_parquet(out_dir / "fold_split.pqt")
eeg_ids_all = index_df.get_column("eeg_id").to_numpy()
np.save(out_dir / "fold_all.npy", eeg_ids_all)
