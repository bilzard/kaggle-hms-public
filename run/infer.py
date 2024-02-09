from pathlib import Path

import hydra
import numpy as np
import polars as pl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.config import MainConfig
from src.constant import LABELS
from src.data_util import preload_cqf, preload_eegs
from src.dataset.eeg import PerEegDataset, get_valid_loader
from src.model.hms_model import HmsModel, check_model
from src.preprocess import process_label
from src.proc_util import trace


def load_metadata(data_dir: Path, phase: str, num_samples: int):
    match phase:
        case "train":
            metadata = pl.read_csv(data_dir / "train.csv")
            metadata = process_label(metadata)
            return (
                metadata.filter(pl.col("duration_sec").eq(50))
                .sample(num_samples, with_replacement=False, seed=42)
                .select("eeg_id", "spectrogram_id", "patient_id")
            )
        case "test":
            return pl.read_csv(data_dir / "test.csv")
        case _:
            raise ValueError(f"Invalid phase: {phase}")


def load_checkpoint(
    model: nn.Module, checkpoint_path: Path, checkpoint_key="checkpoint"
):
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state_dict[checkpoint_key])
    return model


def make_submission(model: nn.Module, test_loader: DataLoader, device: str = "cuda"):
    model.to(device=device)

    test_eeg_ids = []
    test_predictions = []

    for batch in tqdm(test_loader, unit="step"):
        eeg = batch["eeg"].to(device=device)
        cqf = batch["cqf"].to(device=device)
        eeg_id = batch["eeg_id"].detach().cpu().numpy().tolist()
        output = model(dict(eeg=eeg, cqf=cqf))
        pred = output["pred"].softmax(dim=1).detach().cpu().numpy().tolist()
        test_eeg_ids.extend(eeg_id)
        test_predictions.extend(pred)

    test_eeg_ids = np.array(test_eeg_ids)
    test_predictions = np.array(test_predictions)

    submission_df = pl.DataFrame(
        dict(eeg_id=test_eeg_ids)
        | {f"{label}_vote": test_predictions[:, i] for i, label in enumerate(LABELS)}
    )

    return submission_df


@hydra.main(config_path="conf", config_name="baseline", version_base="1.2")
def main(cfg: MainConfig):
    data_dir = Path(cfg.env.data_dir)
    working_dir = Path(cfg.env.working_dir)
    preprocess_dir = Path(working_dir / "preprocess" / cfg.phase / "eeg")

    test_df = load_metadata(data_dir, cfg.phase, cfg.infer.num_samples)
    test_df = process_label(test_df, is_test=True)

    with trace("load eeg"):
        eeg_ids = test_df["eeg_id"].unique().to_list()
        id2eeg = preload_eegs(eeg_ids, preprocess_dir)
        id2cqf = preload_cqf(eeg_ids, preprocess_dir)

    with trace("infer"):
        test_dataset = PerEegDataset(test_df, id2eeg, id2cqf=id2cqf, is_test=True)
        test_loader = get_valid_loader(
            test_dataset,
            batch_size=cfg.infer.batch_size,
            num_workers=cfg.env.num_workers,
            pin_memory=True,
            persistent_workers=False,
        )
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
        submission_df = make_submission(model, test_loader)
        submission_dir = Path(cfg.env.submission_dir)

        submission_df.write_parquet(f"pred_{cfg.phase}.pqt")
        if cfg.final_submission:
            submission_df.write_csv(submission_dir / "submission.csv")
            print(pl.read_csv(submission_dir / "submission.csv"))


if __name__ == "__main__":
    main()
