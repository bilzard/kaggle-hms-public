from pathlib import Path

import hydra
import numpy as np
import polars as pl
import torch
import wandb
from einops import rearrange
from hydra.utils import instantiate
from omegaconf import OmegaConf

from src.callback import MetricsLogger, SaveModelCheckpoint
from src.config import MainConfig
from src.data_util import (
    preload_cqf,
    preload_eegs,
    preload_spectrograms,
    train_valid_split,
)
from src.dataset.eeg import get_train_loader, get_valid_loader
from src.model.hms_model import HmsModel, check_model, get_2d_image
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
    eeg_dir = Path(working_dir / "preprocess" / cfg.phase / "eeg")
    spec_dir = Path(working_dir / "preprocess" / cfg.phase / "spectrogram")

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
        eeg_ids = metadata["eeg_id"].unique().to_list()
        id2eeg = preload_eegs(eeg_ids, eeg_dir)
        id2cqf = preload_cqf(eeg_ids, eeg_dir)

    if cfg.architecture.use_bg_spec:
        with trace("load spectrogram"):
            spec_ids = metadata["spectrogram_id"].unique().to_list()
            spec_id2spec = preload_spectrograms(spec_ids, spec_dir)
    else:
        spec_id2spec = None

    cfg_dict = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    with wandb.init(
        project=cfg.wandb.project,
        name=f"{cfg.exp_name}_fold{cfg.fold}_seed{cfg.seed:04d}",
        config=cfg_dict,  # type: ignore
        mode=cfg.wandb.mode if not cfg.check_only else "disabled",
    ):
        with trace("check model"):
            model = HmsModel(cfg.architecture, pretrained=False)
            model = model.to(device="cuda")
            with torch.no_grad():
                check_model(model, device="cuda")
            del model

        with trace("check 2d image"):
            num_samples = 5
            figure_path = Path("figure")
            if not figure_path.exists():
                figure_path.mkdir(parents=True)

            model = HmsModel(cfg.architecture, pretrained=False)
            model = model.to(device="cuda")
            sample_dataset = instantiate(
                cfg.trainer.train_dataset,
                metadata=train_df.sample(num_samples, shuffle=True, seed=0),
                id2eeg=id2eeg,
                id2cqf=id2cqf,
                spec_id2spec=spec_id2spec,
                duration=cfg.trainer.duration,
                num_samples_per_eeg=cfg.trainer.num_samples_per_eeg,
                transform_enabled=True,
                transform=instantiate(cfg.trainer.transform)
                if cfg.trainer.transform
                else None,
            )
            with torch.no_grad():
                for sample in sample_dataset:
                    eeg_id = sample["eeg_id"]
                    eeg = torch.from_numpy(sample["eeg"][np.newaxis, ...]).to("cuda")
                    cqf = torch.from_numpy(sample["cqf"][np.newaxis, ...]).to("cuda")
                    spec = (
                        torch.from_numpy(sample["spec"][np.newaxis, ...]).to("cuda")
                        if "spec" in sample
                        else None
                    )
                    spec = get_2d_image(model, eeg, cqf, spec, device="cuda")
                    spec = rearrange(spec, "b c f t -> (b c) f t")
                    spec = spec.detach().cpu().numpy().transpose(1, 2, 0)

                    np.save(figure_path / f"spec_{eeg_id}.npy", spec.astype(np.float16))

            del model, sample_dataset
            if cfg.check_only:
                return

        with trace("train model"):
            seed_everything(cfg.seed)
            train_dataset = instantiate(
                cfg.trainer.train_dataset,
                metadata=train_df,
                id2eeg=id2eeg,
                id2cqf=id2cqf,
                spec_id2spec=spec_id2spec,
                duration=cfg.trainer.duration,
                num_samples_per_eeg=cfg.trainer.num_samples_per_eeg,
                transform_enabled=True,
                seed=cfg.seed + cfg.trainer.random_seed_offset,
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
                spec_id2spec=spec_id2spec,
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
