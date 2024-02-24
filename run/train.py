from pathlib import Path

import hydra
import polars as pl
import wandb
from hydra.utils import instantiate
from omegaconf import OmegaConf

from src.callback import MetricsLogger, SaveModelCheckpoint
from src.config import MainConfig
from src.data_util import preload_cqf, preload_eegs, train_valid_split
from src.dataset.eeg import get_train_loader, get_valid_loader
from src.model.hms_model import HmsModel, check_model
from src.preprocess import (
    process_label,
)
from src.proc_util import trace
from src.random_util import seed_everything, seed_worker
from src.sampler import LossBasedSampler
from src.trainer import Trainer


@hydra.main(config_path="conf", config_name="baseline", version_base="1.2")
def main(cfg: MainConfig):
    data_dir = Path(cfg.env.data_dir)
    working_dir = Path(cfg.env.working_dir)
    preprocess_dir = Path(working_dir / "preprocess" / cfg.phase / "eeg")

    metadata = pl.read_csv(data_dir / f"{cfg.phase}.csv")
    metadata = process_label(metadata)

    if cfg.trainer.pseudo_label.enabled:
        teacher_ensemble_name = cfg.trainer.pseudo_label.teacher_ensemble_name
        print(f"* pseudo label {teacher_ensemble_name}")
        loss_df = pl.read_parquet(
            working_dir / "ensemble" / teacher_ensemble_name / "losses.pqt"
        )
        metadata = metadata.join(loss_df, on="eeg_id")
    else:
        metadata = metadata.with_columns(pl.lit(0.0).alias("loss"))

    fold_dir = Path(working_dir / "fold_split" / cfg.phase)
    fold_split_df = pl.read_parquet(fold_dir / "fold_split.pqt")
    train_df, valid_df = train_valid_split(metadata, fold_split_df, fold=cfg.fold)

    print(f"train_df: {train_df.shape}, valid_df: {valid_df.shape}")

    with trace("load eeg"):
        eeg_ids = fold_split_df["eeg_id"].unique().to_list()
        id2eeg = preload_eegs(eeg_ids, preprocess_dir)
        id2cqf = preload_cqf(eeg_ids, preprocess_dir)

    cfg_dict = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    with wandb.init(
        project=cfg.wandb.project,
        name=f"{cfg.exp_name}_fold{cfg.fold}_seed{cfg.seed:04d}",
        config=cfg_dict,  # type: ignore
        mode=cfg.wandb.mode,
    ):
        with trace("check model"):
            model = HmsModel(cfg.architecture)
            check_model(model)
            del model

        with trace("train model"):
            seed_everything(cfg.seed)
            train_dataset = instantiate(
                cfg.trainer.train_dataset,
                metadata=train_df,
                id2eeg=id2eeg,
                id2cqf=id2cqf,
                duration=cfg.trainer.duration,
                num_samples_per_eeg=cfg.trainer.num_samples_per_eeg,
                transform=instantiate(cfg.trainer.transform)
                if cfg.trainer.transform
                else None,
            )
            train_sampler = (
                LossBasedSampler(
                    train_dataset.metadata["loss"].to_numpy(),
                    saturated_epochs=cfg.trainer.pseudo_label.saturated_epochs,
                    initial_sampling_rate=1.0,
                    final_sampling_rate=1 - cfg.trainer.pseudo_label.max_drop_ratio,
                    shuffle=True,
                )
                if cfg.trainer.pseudo_label.enabled
                else None
            )
            train_loader = get_train_loader(
                train_dataset,
                batch_size=cfg.trainer.batch_size,
                num_workers=cfg.env.num_workers,
                pin_memory=True,
                worker_init_fn=seed_worker,
                sampler=train_sampler,
                shuffle=not cfg.trainer.pseudo_label.enabled,
            )
            valid_dataset = instantiate(
                cfg.trainer.valid_dataset,
                metadata=valid_df,
                id2eeg=id2eeg,
                id2cqf=id2cqf,
                duration=cfg.trainer.val.duration,
                stride=cfg.trainer.val.stride,
                seed=cfg.trainer.val.seed,
            )
            valid_loader = get_valid_loader(
                valid_dataset,
                batch_size=cfg.trainer.val.batch_size,
                num_workers=cfg.env.num_workers,
                pin_memory=True,
            )
            model = HmsModel(cfg.architecture)
            model.to(device="cuda")
            trainer = Trainer(
                cfg.trainer,
                model,
                device="cuda",
                train_loader=train_loader,
                valid_loader=valid_loader,
                mixed_precision=True,
                callbacks=[
                    MetricsLogger(aggregation_fn=cfg.trainer.val.aggregation_fn),
                    SaveModelCheckpoint(
                        save_last=cfg.trainer.save_last and not cfg.dry_run,
                        save_best=cfg.trainer.save_best and not cfg.dry_run,
                    ),
                ],
                epochs=cfg.trainer.epochs,
            )
            trainer.fit()


if __name__ == "__main__":
    main()
