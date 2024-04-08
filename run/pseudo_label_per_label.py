from collections import defaultdict
from pathlib import Path
from typing import cast

import hydra
import numpy as np
import polars as pl
import torch
import torch.nn as nn
from hydra.core.global_hydra import GlobalHydra
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.config import EnsembleExperimentConfig, EnsembleMainConfig, MainConfig
from src.constant import LABELS
from src.data_util import preload_cqf, preload_eegs, preload_spectrograms
from src.dataset.eeg import PerLabelDataset, get_valid_loader
from src.infer_util import load_metadata
from src.logger import BaseLogger
from src.proc_util import trace
from src.random_util import seed_everything
from src.train_util import check_model, get_model, move_device

from .infer import load_checkpoint


def load_config(config_name, parent_cfg: EnsembleMainConfig) -> MainConfig:
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
            ],
        )
        print("** phase:", cfg.phase)
        print("** env:", cfg.env.name)
        print("** infer_batch_size:", cfg.infer.batch_size)
        print(
            "** grad_checkpointing:", cfg.architecture.model.encoder.grad_checkpointing
        )

    return cast(MainConfig, cfg)


def get_loader(
    cfg: MainConfig,
    metadata: pl.DataFrame,
    id2eeg: dict[int, np.ndarray],
    id2cqf: dict[int, np.ndarray],
    spec_id2spec: dict[int, np.ndarray],
):
    dataset = PerLabelDataset(
        metadata=metadata,
        id2eeg=id2eeg,
        id2cqf=id2cqf,
        sample_weight=cfg.trainer.data.sampler.sample_weight,
        spec_id2spec=spec_id2spec,
        duration=cfg.trainer.val.duration,
        spec_cropped_duration=cfg.architecture.spec_cropped_duration,
        seed=cfg.trainer.val.seed,
        label_postfix=cfg.trainer.label.label_postfix,
        weight_key=cfg.trainer.label.weight_key,
    )
    data_loader = get_valid_loader(
        dataset,
        batch_size=cfg.trainer.batch_size,
        num_workers=cfg.env.num_workers,
        pin_memory=True,
    )
    return data_loader


def infer_per_seed(
    cfg: MainConfig,
    model: nn.Module,
    data_loader: DataLoader,
    seed: int,
    device: str = "cuda",
) -> pl.DataFrame:
    seed_everything(seed)
    model.to(device)

    label_id2logits = defaultdict(list)

    for batch in tqdm(data_loader, unit="step"):
        with torch.autocast(device_type="cuda", enabled=True):
            move_device(
                batch, cfg.trainer.data.input_keys + ["weight", "label"], "cuda"
            )
            label_ids = batch["label_id"].detach().cpu().numpy()
            output = model(batch)
            logits = output["pred"].detach().cpu().numpy()

            for label_id, logit in zip(label_ids, logits):
                label_id2logits[label_id].append(logit)

    # aggregate per label_id
    for eeg_id, logits in tqdm(label_id2logits.items()):
        # logits: b k c
        logits = np.stack(logits, axis=0)
        assert len(logits.shape) == 3, f"Invalid shape: {logits.shape}"

        logit = logits.mean(axis=(0, 1))
        label_id2logits[eeg_id] = logit

    label_ids = np.array(list(label_id2logits.keys()))
    logits = np.stack(list(label_id2logits.values()), axis=0)

    pseudo_label_df = pl.DataFrame(
        dict(label_id=label_ids)
        | {f"{label}_vote": logits[:, i] for i, label in enumerate(LABELS)}
    )
    return pseudo_label_df


def infer_per_experiment(
    parent_config: EnsembleMainConfig,
    experiment: EnsembleExperimentConfig,
    metadata: pl.DataFrame,
    id2eeg: dict[int, np.ndarray],
    id2cqf: dict[int, np.ndarray],
    spec_id2spec: dict[int, np.ndarray],
) -> pl.DataFrame:
    """
    process per experiment
    1. load config
    2. load data loader
    3. for each fold and seed:
        - load model weight and do inference
    """
    cfg = load_config(experiment.exp_name, parent_config)
    logger = BaseLogger(log_file_name=cfg.infer.log_name, clear=True)
    model = get_model(cfg.architecture, pretrained=False)
    check_model(cfg.architecture, model)

    check_loader = get_loader(
        cfg=cfg,
        metadata=metadata.sample(1),
        id2eeg=id2eeg,
        id2cqf=id2cqf,
        spec_id2spec=spec_id2spec,
    )
    logger.write_log("Dataset:", check_loader.dataset)
    logger.write_log("Model:", model)

    del check_loader

    metadata_all = metadata
    dfs = []
    for ensemble_fold in experiment.folds:
        fold = ensemble_fold.split
        # make PL for OOF samples
        metadata = metadata_all.filter(pl.col("fold").eq(fold))

        data_loader = get_loader(
            cfg=cfg,
            metadata=metadata,
            id2eeg=id2eeg,
            id2cqf=id2cqf,
            spec_id2spec=spec_id2spec,
        )
        for seed in ensemble_fold.seeds:
            print("*" * 50)
            print(f"* exp_name: {cfg.exp_name}, fold: {fold}, seed: {seed}")
            print("*" * 50)
            weight_path = (
                Path(cfg.env.checkpoint_dir)
                / cfg.exp_name
                / f"fold_{fold}"
                / f"seed_{seed}"
                / "model"
                / f"{cfg.infer.model_choice}_model.pth"
            )
            load_checkpoint(model, weight_path)
            pseudo_label_df = infer_per_seed(cfg, model, data_loader, seed)
            dfs.append(pseudo_label_df)

    return pl.concat(dfs)


@hydra.main(config_path="conf", config_name="ensemble", version_base="1.2")
def main(cfg: EnsembleMainConfig):
    data_dir = Path(cfg.env.data_dir)
    working_dir = Path(cfg.env.working_dir)
    eeg_dir = Path(working_dir / "preprocess" / cfg.phase / "eeg")
    spec_dir = Path(working_dir / "preprocess" / cfg.phase / "spectrogram")
    fold_split_dir = Path(working_dir / "fold_split" / cfg.phase)

    metadata = load_metadata(
        data_dir=data_dir,
        phase=cfg.phase,
        fold_split_dir=fold_split_dir,
        num_samples=cfg.dev.num_samples,
        min_weight=0.0,  # load all samples including noisy ones
    )
    metadata = metadata.filter(pl.col("weight").lt(0.3))  # low_qualityのみ抽出
    with trace("** load eeg"):
        eeg_ids = metadata["eeg_id"].unique().to_list()
        id2eeg = preload_eegs(eeg_ids, eeg_dir)
        id2cqf = preload_cqf(eeg_ids, eeg_dir)

    with trace("** load bg spec"):
        spec_ids = metadata["spectrogram_id"].unique().to_list()
        spec_id2spec = preload_spectrograms(spec_ids, spec_dir)

    with trace("** predict per experiments"):
        pred_dfs = []
        for experiment in cfg.ensemble_entity.experiments:
            print(f"*** exp_name: {experiment.exp_name} ***")
            pred_df = infer_per_experiment(
                cfg, experiment, metadata, id2eeg, id2cqf, spec_id2spec
            )
            pred_dfs.append(pred_df)

        pred_df = (
            pl.concat(pred_dfs)
            .group_by("label_id", maintain_order=True)
            .agg(
                *[pl.col(f"{label}_vote").mean() for label in LABELS],
            )
        )
        pred_df = pred_df.rename(
            {f"{label}_vote": f"pl_{label}_vote" for label in LABELS}
        )

    with trace("** generate pseudo label"):
        pred_df.write_parquet(f"{cfg.phase}_pseudo_label.pqt")
        print(pred_df)


if __name__ == "__main__":
    main()
