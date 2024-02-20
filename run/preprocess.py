import shutil
from pathlib import Path

import hydra
import numpy as np
import polars as pl
from tqdm import tqdm

from src.config import MainConfig
from src.constant import EEG_PROBES, PROBES
from src.preprocess import (
    load_eeg,
    process_cqf,
    process_eeg,
    process_label,
    select_develop_samples,
)
from src.proc_util import trace


def save_eeg(eeg_id: str, eeg_df: pl.DataFrame, output_dir: Path):
    eeg = eeg_df.select(PROBES).to_numpy()

    output_file_path = output_dir / str(eeg_id)
    if not output_file_path.exists():
        output_file_path.mkdir(parents=True)

    info = np.finfo(np.float16)
    eeg = eeg.clip(info.min, info.max).astype(np.float16)
    np.save(output_file_path / "eeg.npy", eeg)


def save_pad_mask(eeg_id: str, pad_mask: np.ndarray, output_dir: Path):
    pad_mask = pad_mask.astype(np.uint8)
    np.save(output_dir / str(eeg_id) / "pad_mask.npy", pad_mask)


def save_cqf(eeg_id: str, eeg_df: pl.DataFrame, output_dir: Path):
    cqf = (
        eeg_df.select(f"CQF-{probe}" for probe in EEG_PROBES)
        .to_numpy()
        .astype(np.float16)
    )
    mask = (
        eeg_df.select(f"mask-{probe}" for probe in EEG_PROBES)
        .to_numpy()
        .astype(np.uint8)
    )

    output_file_path = output_dir / str(eeg_id)
    if not output_file_path.exists():
        output_file_path.mkdir(parents=True)

    np.save(output_file_path / "cqf.npy", cqf)
    np.save(output_file_path / "mask.npy", mask)


@hydra.main(config_path="conf", config_name="main", version_base="1.2")
def main(cfg: MainConfig):
    real_phase = cfg.phase if cfg.phase != "develop" else "train"

    data_dir = Path(cfg.env.data_dir)

    metadata = pl.read_csv(data_dir / f"{real_phase}.csv")
    if cfg.phase == "develop":
        columns_org = metadata.columns
        metadata = process_label(metadata)
        metadata = select_develop_samples(metadata)
        metadata = metadata.select(columns_org)

    output_dir_eeg = Path("eeg")

    if (not cfg.dry_run) and (cfg.cleanup) and (output_dir_eeg.exists()):
        shutil.rmtree(output_dir_eeg)
        print(f"Removed {cfg.phase} dir: {output_dir_eeg}")
    else:
        output_dir_eeg.mkdir(parents=True, exist_ok=True)
        print(f"Created {cfg.phase} dir: {output_dir_eeg}")

    eeg_ids = metadata["eeg_id"].unique().to_numpy()
    if cfg.debug:
        num_samples = 100
        eeg_ids = eeg_ids[: min(len(eeg_ids), num_samples)]

    tag = " (with cqf)" if cfg.preprocess.process_cqf else ""

    with trace(f"process eeg{tag}"):
        for eeg_id in tqdm(eeg_ids, total=eeg_ids.shape[0]):
            eeg_df = load_eeg(eeg_id, data_dir=data_dir, phase=real_phase)
            eeg, pad_mask = process_eeg(eeg_df)

            eeg /= cfg.preprocess.ref_voltage
            eeg_df = pl.DataFrame(
                {probe: pl.Series(v) for probe, v in zip(PROBES, np.transpose(eeg))}
            )

            if cfg.preprocess.process_cqf:
                eeg_df = process_cqf(eeg_df)

            if not cfg.dry_run:
                save_eeg(eeg_id, eeg_df, output_dir_eeg)
                save_pad_mask(eeg_id, pad_mask, output_dir_eeg)

                if cfg.preprocess.process_cqf:
                    save_cqf(eeg_id, eeg_df, output_dir_eeg)


if __name__ == "__main__":
    main()
