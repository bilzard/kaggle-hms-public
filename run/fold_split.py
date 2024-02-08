from pathlib import Path

import hydra
import numpy as np
import polars as pl
from sklearn.model_selection import GroupKFold

from src.config import MainConfig
from src.preprocess import process_label
from src.proc_util import trace


def split_group_k_fold(metadata: pl.DataFrame, num_splits: int) -> pl.DataFrame:
    """
    patient_idをgroupとして、group k-foldで分割する。
    """
    group_kfold = GroupKFold(n_splits=num_splits)
    df = metadata.select("eeg_id", "patient_id").unique(maintain_order=True).to_pandas()

    df["fold"] = -1
    for fold, (_, valid_idxs) in enumerate(
        group_kfold.split(df[["eeg_id"]], df["eeg_id"], df["patient_id"])
    ):
        df.loc[valid_idxs, "fold"] = fold
    df["fold"] = df["fold"].astype(np.int8)

    return pl.from_pandas(df)


def split_single_label(
    metadata: pl.DataFrame, num_validation_patients: int
) -> pl.DataFrame:
    """
    LBと整合させるため、validationにはsingle labelのEEGが大半を占めるようにする。
    validation/trainの分割方法は以下の通り:

    1. single labelのEEGを持つ患者のうち、EEGあたりのラベル数の最大値を小さい順にソート
    2. 1のうち、先頭の num_validation_patients 件を抽出する
    3. 2のpatient_idを持つEEGをvalidation setに含める。それ以外のEEGはtrain setに含める
    """
    eeg_df = (
        metadata.group_by("eeg_id", maintain_order=True)
        .agg(
            pl.col("patient_id").first(),
            pl.col("duration_sec").first(),
            pl.col("num_labels_per_eeg").first(),
        )
        .with_columns(
            pl.col("num_labels_per_eeg").eq(1).alias("is_single_label"),
        )
    )
    patient_df = (
        eeg_df.group_by("patient_id", maintain_order=True)
        .agg(
            pl.col("num_labels_per_eeg").max().alias("max_num_labels_per_eeg"),
            pl.col("num_labels_per_eeg").min().alias("min_num_labels_per_eeg"),
        )
        .with_columns(
            pl.col("min_num_labels_per_eeg").eq(1).alias("has_single_label_eeg"),
        )
        .sort("max_num_labels_per_eeg")
    )
    validation_patients = patient_df.filter(pl.col("has_single_label_eeg")).head(
        num_validation_patients
    )
    validation_patient_ids = set(validation_patients["patient_id"].to_numpy().tolist())
    eeg_df = eeg_df.with_columns(
        pl.when(pl.col("patient_id").is_in(validation_patient_ids))
        .then(pl.lit("validation"))
        .otherwise(pl.lit("train"))
        .alias("fold")
    )
    return eeg_df.select("eeg_id", "patient_id", "fold")


@hydra.main(config_path="conf", config_name="main", version_base="1.2")
def main(cfg: MainConfig):
    data_dir = Path(cfg.env.data_dir)
    metadata = pl.read_csv(data_dir / f"{cfg.phase}.csv")
    metadata = process_label(metadata)

    with trace("split fold"):
        if cfg.split.strategy == "group_k_fold":
            df = split_group_k_fold(metadata, cfg.split.num_splits)
        elif cfg.split.strategy == "single_label":
            df = split_single_label(metadata, cfg.split.num_validation_patients)
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
