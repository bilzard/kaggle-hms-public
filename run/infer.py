from collections import defaultdict
from pathlib import Path

import hydra
import numpy as np
import polars as pl
import torch
import torch.nn as nn
from hydra.utils import instantiate
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.config import MainConfig
from src.data_util import preload_cqf, preload_eegs, preload_spectrograms
from src.dataset.eeg import get_valid_loader
from src.evaluator import Evaluator
from src.infer_util import load_metadata, make_submission
from src.logger import BaseLogger
from src.proc_util import trace
from src.random_util import seed_everything, seed_worker
from src.train_util import check_model, get_model, move_device


def load_checkpoint(
    model: nn.Module, checkpoint_path: Path, checkpoint_key="checkpoint"
):
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state_dict[checkpoint_key])
    return model


def get_loader(
    cfg: MainConfig,
    metadata: pl.DataFrame,
    id2eeg: dict[int, np.ndarray],
    id2cqf: dict[int, np.ndarray],
    spec_id2spec: dict[int, np.ndarray] | None = None,
) -> DataLoader:
    match cfg.phase:
        case "train":
            valid_dataset = instantiate(
                cfg.trainer.valid_dataset,
                metadata=metadata,
                id2eeg=id2eeg,
                id2cqf=id2cqf,
                spec_id2spec=spec_id2spec,
                duration=cfg.trainer.val.duration,
                stride=cfg.trainer.val.stride,
                seed=cfg.trainer.val.seed,
                spec_cropped_duration=cfg.architecture.spec_cropped_duration,
                transform_enabled=True,
                transform=instantiate(cfg.infer.tta)
                if cfg.infer.tta is not None
                else None,
                label_postfix=cfg.trainer.label.label_postfix,
                weight_key=cfg.trainer.label.weight_key,
            )
            valid_loader = get_valid_loader(
                valid_dataset,
                batch_size=cfg.trainer.val.batch_size,
                num_workers=cfg.env.num_workers,
                worker_init_fn=seed_worker,
                pin_memory=True,
                persistent_workers=False,
            )
            return valid_loader
        case "test" | "develop":
            test_dataset = instantiate(
                cfg.infer.test_dataset,
                metadata=metadata,
                id2eeg=id2eeg,
                id2cqf=id2cqf,
                spec_id2spec=spec_id2spec,
                with_label=False,
                spec_cropped_duration=cfg.architecture.spec_cropped_duration,
                duration=cfg.infer.test_dataset.duration,
                transform_enabled=True,
                transform=instantiate(cfg.infer.tta)
                if cfg.infer.tta is not None
                else None,
            )
            test_loader = get_valid_loader(
                test_dataset,
                batch_size=cfg.infer.batch_size,
                num_workers=cfg.env.num_workers,
                pin_memory=True,
            )
            return test_loader
        case _:
            raise ValueError(f"Invalid phase: {cfg.phase}")


def predict(
    model: nn.Module,
    test_loader: DataLoader,
    input_keys: list[str],
    iterations: int = 1,
    device: str = "cuda",
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    model.to(device=device)

    eeg_id2logits = defaultdict(list)

    for i in range(iterations):
        print(f"iteration: {i+1}/{iterations}")
        for batch in tqdm(test_loader, unit="step"):
            with torch.autocast(device_type="cuda", enabled=True):
                move_device(batch, input_keys, device)
                eeg_ids = batch["eeg_id"].detach().cpu().numpy()
                output = model(batch)
                logits = output["pred"].detach().cpu().numpy()

                for eeg_id, logit in zip(eeg_ids, logits):
                    # NOTE: このif文の意図は...? TTAしても最初のやつしか考慮されないような...?
                    if eeg_id not in eeg_id2logits:
                        eeg_id2logits[eeg_id].append(logit)

    # aggregate per EEG ID
    for eeg_id, logits in tqdm(eeg_id2logits.items()):
        # logits: b k c
        logits = np.stack(logits, axis=0)
        assert len(logits.shape) == 3, f"Invalid shape: {logits.shape}"

        logit = logits.mean(axis=(0, 1))
        eeg_id2logits[eeg_id] = logit

    eeg_ids = np.array(list(eeg_id2logits.keys()))
    logits = np.stack(list(eeg_id2logits.values()), axis=0)

    return eeg_ids, logits


@hydra.main(config_path="conf", config_name="baseline", version_base="1.2")
def main(cfg: MainConfig):
    data_dir = Path(cfg.env.data_dir)
    working_dir = Path(cfg.env.working_dir)
    eeg_dir = Path(working_dir / "preprocess" / cfg.phase / "eeg")
    spec_dir = Path(working_dir / "preprocess" / cfg.phase / "spectrogram")
    fold_split_dir = Path(working_dir / "fold_split" / cfg.phase)

    metadata = load_metadata(
        data_dir=data_dir,
        phase=cfg.phase,
        fold_split_dir=fold_split_dir,
        fold=cfg.fold,
        num_samples=cfg.dev.num_samples,
        min_weight=cfg.trainer.val.min_weight,
    )

    logger = BaseLogger(log_file_name=cfg.infer.log_name, clear=True)

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

    with trace("infer"):
        data_loader = get_loader(
            cfg, metadata, id2eeg, id2cqf, spec_id2spec=spec_id2spec
        )

        model = get_model(cfg.architecture, pretrained=False)
        check_model(cfg.architecture, model)
        logger.write_log("Dataset:", data_loader.dataset)
        logger.write_log("Model:", model)
        if cfg.check_only:
            return

        weight_path = (
            Path(cfg.env.checkpoint_dir)
            / cfg.exp_name
            / f"fold_{cfg.fold}"
            / f"seed_{cfg.seed}"
            / "model"
            / f"{cfg.infer.model_choice}_model.pth"
        )
        load_checkpoint(model, weight_path)

        seed_everything(cfg.seed)
        match cfg.phase:
            case "train":
                evaluator = Evaluator(
                    aggregation_fn=cfg.trainer.val.aggregation_fn,
                    input_keys=cfg.trainer.data.input_keys,
                    agg_policy=cfg.trainer.val.agg_policy,
                    iterations=cfg.infer.tta_iterations,
                    weight_exponent=cfg.trainer.val.weight_exponent,
                    min_weight=cfg.trainer.val.min_weight,
                )
                logger.write_log("Evaluator:", evaluator)
                output = evaluator.evaluate(model, data_loader)
                val_loss, val_loss_per_label, eeg_ids, logits = (
                    output["val_loss"],
                    output["val_loss_per_label"],
                    output["eeg_ids"],
                    output["logits_per_eeg"],
                )
                print(f"val_loss: {val_loss:.4f}")
                print(
                    ", ".join([f"{k}={v:.4f}" for k, v in val_loss_per_label.items()])
                )
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
        prediction_df.write_parquet(f"pred_{cfg.phase}.pqt")

        if cfg.final_submission:
            submission_df = make_submission(eeg_ids, logits, apply_softmax=True)
            submission_dir = Path(cfg.env.submission_dir)
            submission_df.write_csv(submission_dir / "submission.csv")
            print(pl.read_csv(submission_dir / "submission.csv"))


if __name__ == "__main__":
    main()
