from pathlib import Path
from typing import cast

import hydra
import numpy as np
import polars as pl
import torch.nn as nn
from hydra.core.global_hydra import GlobalHydra
from torch.utils.data import DataLoader

from src.config import EnsembleExperimentConfig, EnsembleMainConfig, MainConfig
from src.constant import LABELS
from src.data_util import preload_cqf, preload_eegs, preload_spectrograms
from src.evaluator import Evaluator
from src.infer_util import load_metadata, make_submission
from src.logger import BaseLogger
from src.proc_util import trace
from src.random_util import seed_everything
from src.train_util import check_model, get_model

from .ensemble import do_evaluate
from .infer import get_loader, load_checkpoint, predict


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


def infer_per_seed(
    cfg: MainConfig, model: nn.Module, data_loader: DataLoader, seed: int
) -> pl.DataFrame:
    # TODO: 各fold/seedごとの推論結果をparquetでinferディレクトリ配下に保存する
    seed_everything(seed)

    match cfg.phase:
        # TODO: evaluateの有無をオプションで指定できるようにする
        case "train":
            evaluator = Evaluator(
                aggregation_fn=cfg.trainer.val.aggregation_fn,
                input_keys=cfg.trainer.data.input_keys,
                agg_policy=cfg.trainer.val.agg_policy,
                iterations=cfg.infer.tta_iterations,
            )
            val_loss, val_loss_per_label, eeg_ids, logits = evaluator.evaluate(
                model, data_loader
            )
            print(f"val_loss: {val_loss:.4f}")
            print(", ".join([f"{k}={v:.4f}" for k, v in val_loss_per_label.items()]))
        case "test" | "develop":
            eeg_ids, logits = predict(
                model,
                data_loader,
                cfg.trainer.data.input_keys,
                iterations=cfg.infer.tta_iterations,
            )
        case _:
            raise ValueError(f"Invalid phase: {cfg.phase}")

    prediction_df = make_submission(eeg_ids, logits, apply_softmax=False)
    return prediction_df


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
    pred_dfs = []
    for ensemble_fold in experiment.folds:
        fold = ensemble_fold.split
        # trainの場合、validation dataのみ推論する
        if cfg.phase == "train":
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
            pred_df = infer_per_seed(cfg, model, data_loader, seed)
            pred_dfs.append(pred_df)

    return pl.concat(pred_dfs)


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
    )
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
            .group_by("eeg_id", maintain_order=True)
            .agg(pl.col(f"{label}_vote").mean() for label in LABELS)
        )

    with trace("** evaluate or make submission"):
        match cfg.phase:
            case "train":
                metadata = load_metadata(
                    data_dir=data_dir,
                    phase=cfg.phase,
                    fold_split_dir=fold_split_dir,
                    group_by_eeg=True,
                    weight_key="weight_per_eeg",
                    num_samples=cfg.dev.num_samples,
                )
                do_evaluate(metadata, pred_df)

            case "test":
                submission_df = make_submission(
                    eeg_ids=pred_df["eeg_id"].to_list(),
                    predictions=pred_df.drop("eeg_id").to_numpy(),
                    apply_softmax=True,
                )
                submission_dir = Path(cfg.env.submission_dir)
                submission_df.write_csv(submission_dir / "submission.csv")
                print(submission_df)


if __name__ == "__main__":
    main()
