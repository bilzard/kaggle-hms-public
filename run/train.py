import shutil
from pathlib import Path
from typing import cast

import hydra
import numpy as np
import polars as pl
import torch
import torch.nn as nn
import wandb
from einops import rearrange
from hydra.core.global_hydra import GlobalHydra
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
from src.preprocess import (
    process_label,
)
from src.proc_util import trace
from src.random_util import seed_everything, seed_worker
from src.sampler import LossBasedSampler
from src.train_util import check_model, get_model, move_device


def load_checkpoint(
    model: nn.Module, checkpoint_path: Path, checkpoint_key="checkpoint"
):
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state_dict[checkpoint_key])
    return model


def load_config(config_name: str, parent_cfg: MainConfig) -> MainConfig:
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()

    with hydra.initialize(config_path="conf", version_base="1.2"):
        cfg = hydra.compose(
            config_name=config_name,
            overrides=[
                f"phase={parent_cfg.phase}",
                f"env={parent_cfg.env.name}",
                f"infer.batch_size={parent_cfg.env.infer_batch_size}",
                f"architecture.model.encoder.grad_checkpointing={parent_cfg.env.grad_checkpointing}",
                f"fold={parent_cfg.fold}",
                f"seed={parent_cfg.seed}",
            ],
        )
        print("** phase:", cfg.phase)
        print("** env:", cfg.env.name)
        print("** infer_batch_size:", cfg.infer.batch_size)
        print(
            "** grad_checkpointing:", cfg.architecture.model.encoder.grad_checkpointing
        )

    return cast(MainConfig, cfg)


@torch.no_grad()
def save_sample_spec(
    cfg: MainConfig,
    metadata: pl.DataFrame,
    id2eeg: dict[str, np.ndarray],
    id2cqf: dict[str, np.ndarray],
    spec_id2spec: dict[str, np.ndarray] | None = None,
    num_samples: int = 5,
    seed: int = 0,
    device: str = "cuda",
    figure_path: Path = Path("figure"),
    cleanup: bool = True,
):
    seed_everything(seed)
    if not figure_path.exists():
        if cleanup:
            shutil.rmtree(figure_path, ignore_errors=True)
        figure_path.mkdir(parents=True)

    model = get_model(cfg.architecture, pretrained=False)
    model = model.to(device=device)
    model.train()

    sample_dataset = instantiate(
        cfg.trainer.train_dataset,
        metadata=metadata.sample(
            min(num_samples, len(metadata)), shuffle=True, seed=seed
        ),
        id2eeg=id2eeg,
        id2cqf=id2cqf,
        spec_id2spec=spec_id2spec,
        duration=cfg.trainer.duration,
        num_samples_per_eeg=cfg.trainer.num_samples_per_eeg,
        spec_cropped_duration=cfg.architecture.spec_cropped_duration,
        transform_enabled=True,
        transform=instantiate(cfg.trainer.transform) if cfg.trainer.transform else None,
    )
    eeg_ids = [sample["eeg_id"] for sample in sample_dataset]
    eeg = torch.stack([torch.from_numpy(sample["eeg"]) for sample in sample_dataset])
    cqf = torch.stack([torch.from_numpy(sample["cqf"]) for sample in sample_dataset])
    label = torch.stack(
        [torch.from_numpy(sample["label"]) for sample in sample_dataset]
    )
    weight = torch.stack(
        [torch.from_numpy(sample["weight"]) for sample in sample_dataset]
    )
    bg_spec = (
        torch.stack([torch.from_numpy(sample["bg_spec"]) for sample in sample_dataset])
        if cfg.architecture.use_bg_spec
        else None
    )
    batch = dict(eeg=eeg, cqf=cqf, label=label, weight=weight)
    if bg_spec is not None:
        batch |= dict(bg_spec=bg_spec)

    input_keys = cfg.trainer.data.input_keys + ["label", "weight"]
    move_device(batch, input_keys=input_keys, device=device)

    if not hasattr(model, "preprocess"):
        print("model does not have preprocess method. skip generating sample image.")
        return

    output = model.preprocess(batch)
    eegs = output.get("eeg", None)
    specs = output.get("spec", None)

    if eegs is not None:
        eegs = rearrange(eegs, "(d b) (c ch) t -> b ch t (d c)", d=2, c=2)
        eegs = eegs.detach().cpu().numpy()
        for eeg_id, eeg in zip(eeg_ids, eegs):
            np.save(figure_path / f"eeg_{eeg_id}.npy", eeg.astype(np.float16))

    if specs is not None:
        d = specs.shape[0] // num_samples
        specs = rearrange(specs, "(d b) c f t -> b f t (d c)", d=d, b=num_samples)
        specs = specs.detach().cpu().numpy()
        for eeg_id, spec in zip(eeg_ids, specs):
            np.save(figure_path / f"spec_{eeg_id}.npy", spec.astype(np.float16))


@hydra.main(config_path="conf", config_name="baseline", version_base="1.2")
def main(cfg: MainConfig):
    data_dir = Path(cfg.env.data_dir)
    working_dir = Path(cfg.env.working_dir)
    eeg_dir = Path(working_dir / "preprocess" / cfg.phase / "eeg")
    spec_dir = Path(working_dir / "preprocess" / cfg.phase / "spectrogram")

    metadata = pl.read_csv(data_dir / f"{cfg.phase}.csv")
    metadata = process_label(
        metadata,
        population_power=cfg.trainer.label.population_power,
        diversity_power=cfg.trainer.label.diversity_power,
        max_votes=cfg.trainer.label.max_votes,
    )

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
    train_df = train_df.filter(
        pl.col("population").gt(cfg.trainer.population_threshold)
    )
    valid_df = valid_df.filter(
        pl.col("population").gt(cfg.trainer.val.population_threshold)
    )

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
        print(f"train_df: {train_df.shape}, valid_df: {valid_df.shape}")
        with trace("check model"):
            model = get_model(cfg.architecture, pretrained=False)
            model = model.to(device="cuda")
            if cfg.debug:
                print(model)
            with torch.no_grad():
                check_model(cfg.architecture, model, device="cuda")
            del model

        with trace("save sample images"):
            save_sample_spec(
                cfg,
                train_df,
                id2eeg,
                id2cqf,
                spec_id2spec,
                num_samples=cfg.trainer.batch_size,
                figure_path=Path("figure"),
            )

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
                spec_cropped_duration=cfg.architecture.spec_cropped_duration,
                transform_enabled=True,
                seed=cfg.seed + cfg.trainer.random_seed_offset,
                transform=instantiate(cfg.trainer.transform)
                if cfg.trainer.transform
                else None,
                label_postfix=cfg.trainer.label.label_postfix,
                weight_key=cfg.trainer.label.weight_key,
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
                spec_cropped_duration=cfg.architecture.spec_cropped_duration,
                label_postfix=cfg.trainer.label.label_postfix,
                weight_key=cfg.trainer.label.weight_key,
            )
            valid_loader = get_valid_loader(
                valid_dataset,
                batch_size=cfg.trainer.val.batch_size,
                num_workers=cfg.env.num_workers,
                pin_memory=True,
            )

            if cfg.trainer.distillation.teacher_exp_name:
                teacher_cfg = load_config(
                    cfg.trainer.distillation.teacher_exp_name, cfg
                )
                teacher_model = get_model(teacher_cfg.architecture, pretrained=False)
                weight_path = (
                    Path(cfg.env.checkpoint_dir)
                    / cfg.trainer.distillation.teacher_exp_name
                    / f"fold_{cfg.fold}"
                    / f"seed_{cfg.trainer.distillation.teacher_seed}"
                    / "model"
                    / f"{teacher_cfg.infer.model_choice}_model.pth"
                )
                load_checkpoint(teacher_model, weight_path)
                teacher_model.to(device="cuda")
            else:
                teacher_model = None

            model = get_model(cfg.architecture)
            model.to(device="cuda")
            trainer = instantiate(
                cfg.trainer.trainer_class,
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
                teacher_model=teacher_model,
            )
            trainer.fit()


if __name__ == "__main__":
    main()
