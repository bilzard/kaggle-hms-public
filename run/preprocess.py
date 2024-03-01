from pathlib import Path

import hydra
import numpy as np
import polars as pl
from tqdm import tqdm

from src.config import MainConfig
from src.constant import EEG_PROBES, PROBES
from src.preprocess import (
    load_eeg,
    load_spectrogram,
    process_cqf,
    process_eeg,
    process_label,
    process_spectrogram,
    select_develop_samples,
)
from src.proc_util import trace


def mkdir_if_not_exists(dir_path: Path):
    if not dir_path.exists():
        dir_path.mkdir(parents=True, exist_ok=True)


def save_eeg(eeg_id: str, eeg_df: pl.DataFrame, output_dir: Path):
    eeg = eeg_df.select(PROBES).to_numpy()
    output_file_path = output_dir / str(eeg_id)
    mkdir_if_not_exists(output_file_path)

    info = np.finfo(np.float16)
    eeg = eeg.clip(info.min, info.max).astype(np.float16)
    np.save(output_file_path / "eeg.npy", eeg)


def save_spectrogram(
    spectrogram_id: str, spec: np.ndarray, output_dir: Path, dtype=np.float16
):
    output_file_path = output_dir / str(spectrogram_id)
    mkdir_if_not_exists(output_file_path)
    spec = spec.astype(dtype)
    np.save(output_dir / str(spectrogram_id) / "spectrogram.npy", spec)


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
    output_dir_spectrogram = Path("spectrogram")

    if output_dir_eeg.exists():
        print(f"The directory {output_dir_eeg} already exists. Skip preprocessing eeg.")
    else:
        mkdir_if_not_exists(output_dir_eeg)

        with trace(f"process eeg (process_cqf={cfg.preprocess.process_cqf})"):
            eeg_ids = metadata["eeg_id"].unique().to_numpy()
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

    if output_dir_spectrogram.exists():
        print(
            f"The directory {output_dir_spectrogram} already exists. Skip processing spectrogram."
        )
    else:
        mkdir_if_not_exists(output_dir_spectrogram)
        with trace("process spectrogram"):
            spectrogram_ids = (
                metadata["spectrogram_id"].unique(maintain_order=True).to_numpy()
            )
            for spectrogram_id in tqdm(spectrogram_ids, total=spectrogram_ids.shape[0]):
                spectrogram_df = load_spectrogram(
                    spectrogram_id, data_dir=data_dir, phase=real_phase
                )
                spectrogram = process_spectrogram(spectrogram_df)

                if not cfg.dry_run:
                    save_spectrogram(
                        spectrogram_id, spectrogram, output_dir_spectrogram
                    )


if __name__ == "__main__":
    main()
