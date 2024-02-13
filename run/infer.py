from pathlib import Path

import hydra
import numpy as np
import polars as pl
import torch
import torch.nn as nn
from scipy.special import softmax
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.config import MainConfig
from src.constant import LABELS
from src.data_util import preload_cqf, preload_eegs
from src.dataset.eeg import PerEegDataset, SlidingWindowEegDataset, get_valid_loader
from src.evaluator import Evaluator
from src.model.hms_model import HmsModel, check_model
from src.preprocess import process_label
from src.proc_util import trace


def load_metadata(
    data_dir: Path, phase: str, num_samples: int, fold_split_dir: Path, fold: int = -1
) -> pl.DataFrame:
    """
    phaseに応じてmetadataをロードする
    train:
        - validationのeegのみをロード
        - eeg_idは重複あり
    test:
        - 全てのeegをロード
        - eeg_idは重複なし
    """
    match phase:
        case "train":
            fold_split_df = pl.read_parquet(fold_split_dir / "fold_split.pqt")
            eeg_ids = (
                fold_split_df.filter(pl.col("fold").eq(fold)).select("eeg_id").unique()
            )
            metadata = pl.read_csv(data_dir / "train.csv")
            metadata = metadata.join(eeg_ids, on="eeg_id")
            metadata = process_label(metadata)

            return metadata
        case "test":
            metadata = pl.read_csv(data_dir / "test.csv")
            metadata = process_label(metadata, is_test=True)
            return metadata
        case _:
            raise ValueError(f"Invalid phase: {phase}")


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
) -> DataLoader:
    match cfg.phase:
        case "train":
            valid_dataset = SlidingWindowEegDataset(
                metadata,
                id2eeg,
                id2cqf=id2cqf,
                duration=cfg.trainer.val.duration,
                stride=cfg.trainer.val.stride,
            )
            valid_loader = get_valid_loader(
                valid_dataset,
                batch_size=cfg.trainer.val.batch_size,
                num_workers=cfg.env.num_workers,
                pin_memory=True,
            )
            return valid_loader
        case "test":
            test_dataset = PerEegDataset(metadata, id2eeg, id2cqf=id2cqf, is_test=True)
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
    model: nn.Module, test_loader: DataLoader, device: str = "cuda"
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    model.to(device=device)

    eeg_ids = []
    logits = []

    for batch in tqdm(test_loader, unit="step"):
        eeg = batch["eeg"].to(device=device)
        cqf = batch["cqf"].to(device=device)
        eeg_id = batch["eeg_id"].detach().cpu().numpy().tolist()
        output = model(dict(eeg=eeg, cqf=cqf))
        logit = output["pred"].detach().cpu().numpy().tolist()
        eeg_ids.extend(eeg_id)
        logits.extend(logit)

    eeg_ids = np.array(eeg_ids)
    logits = np.array(logits)
    return eeg_ids, logits


def make_submission(eeg_ids, predictions, apply_softmax=True):
    if apply_softmax:
        predictions = softmax(predictions, axis=1)
    submission_df = pl.DataFrame(
        dict(eeg_id=eeg_ids)
        | {f"{label}_vote": predictions[:, i] for i, label in enumerate(LABELS)}
    )
    return submission_df


@hydra.main(config_path="conf", config_name="baseline", version_base="1.2")
def main(cfg: MainConfig):
    data_dir = Path(cfg.env.data_dir)
    working_dir = Path(cfg.env.working_dir)
    preprocess_dir = Path(working_dir / "preprocess" / cfg.phase / "eeg")
    fold_split_dir = Path(working_dir / "fold_split" / cfg.phase)

    metadata = load_metadata(
        data_dir, cfg.phase, cfg.infer.num_samples, fold_split_dir, cfg.fold
    )

    with trace("load eeg"):
        eeg_ids = metadata["eeg_id"].unique().to_list()
        id2eeg = preload_eegs(eeg_ids, preprocess_dir)
        id2cqf = preload_cqf(eeg_ids, preprocess_dir)

    with trace("infer"):
        data_loader = get_loader(cfg, metadata, id2eeg, id2cqf)
        model = HmsModel(cfg.architecture, pretrained=False)
        check_model(model)
        weight_path = (
            Path(cfg.env.checkpoint_dir)
            / cfg.exp_name
            / f"fold_{cfg.fold}"
            / f"seed_{cfg.seed}"
            / "model"
            / f"{cfg.infer.model_choice}_model.pth"
        )
        load_checkpoint(model, weight_path)

        match cfg.phase:
            case "train":
                evaluator = Evaluator(aggregation_fn=cfg.trainer.val.aggregation_fn)
                val_loss, val_loss_per_label, eeg_ids, logits = evaluator.evaluate(
                    model, data_loader
                )
                print(f"val_loss: {val_loss:.4f}")
                print(
                    ", ".join([f"{k}={v:.4f}" for k, v in val_loss_per_label.items()])
                )
            case "test":
                eeg_ids, logits = predict(model, data_loader)
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
