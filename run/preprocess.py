import warnings
from functools import partial
from multiprocessing import get_context
from pathlib import Path

import hydra
import numpy as np
import polars as pl
from tqdm import tqdm

from src.config import MainConfig
from src.constant import EEG_PROBES, PROBES
from src.preprocess import (
    do_process_cqf,
    load_eeg,
    load_spectrogram,
    process_eeg,
    process_label,
    process_spectrogram,
    select_develop_samples,
)
from src.proc_util import trace

warnings.filterwarnings("ignore", message="Mean of empty slice")


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


def process_single_eeg(
    eeg_id: int,
    data_dir: Path,
    output_dir: Path,
    phase: str,
    clip_val: float,
    ref_voltage: float,
    process_cqf: bool,
    apply_filter: bool,
    cutoff_freqs: tuple[float | None, float | None],
    reject_freq: float | None,
    device: str,
    dry_run: bool = False,
) -> None:
    eeg_df = load_eeg(eeg_id, data_dir=data_dir, phase=phase)
    eeg, pad_mask = process_eeg(
        eeg_df,
        clip_val=clip_val,
        apply_filter=apply_filter,
        cutoff_freqs=cutoff_freqs,
        reject_freq=reject_freq,
        device=device,
    )

    eeg /= ref_voltage
    eeg_df = pl.DataFrame(
        {probe: pl.Series(v) for probe, v in zip(PROBES, np.transpose(eeg))}
    )
    if process_cqf:
        eeg_df = do_process_cqf(eeg_df)

    if not dry_run:
        save_eeg(str(eeg_id), eeg_df, output_dir)
        save_pad_mask(str(eeg_id), pad_mask, output_dir)

        if process_cqf:
            save_cqf(str(eeg_id), eeg_df, output_dir)


def preprocess_eeg(
    metadata: pl.DataFrame,
    cfg: MainConfig,
    data_dir: Path,
    output_dir: Path,
    phase: str,
):
    if output_dir.exists() and not cfg.cleanup:
        print(f"The directory {output_dir} already exists. Skip preprocessing eeg.")
        return

    mkdir_if_not_exists(output_dir)

    with trace(f"process eeg (process_cqf={cfg.preprocess.process_cqf})"):
        eeg_ids = metadata["eeg_id"].unique().to_numpy()
        process_fn = partial(
            process_single_eeg,
            data_dir=data_dir,
            output_dir=output_dir,
            phase=phase,
            clip_val=cfg.preprocess.clip_val,
            ref_voltage=cfg.preprocess.ref_voltage,
            process_cqf=cfg.preprocess.process_cqf,
            apply_filter=cfg.preprocess.apply_filter,
            cutoff_freqs=cfg.preprocess.cutoff_freqs,
            reject_freq=cfg.preprocess.reject_freq,
            device=cfg.preprocess.device,
            dry_run=cfg.dry_run,
        )
        with get_context("spawn").Pool(cfg.env.num_workers) as pool:
            print(f"Start processing eeg with {cfg.env.num_workers} workers.")
            list(tqdm(pool.imap(process_fn, eeg_ids), total=eeg_ids.shape[0]))


def process_single_spectrogram(
    spectrogram_id: int, data_dir: Path, phase: str, output_dir: Path, dry_run: bool
) -> None:
    spectrogram_df = load_spectrogram(spectrogram_id, data_dir=data_dir, phase=phase)
    spectrogram = process_spectrogram(spectrogram_df)

    if not dry_run:
        save_spectrogram(str(spectrogram_id), spectrogram, output_dir)


def preprocess_spectrogram(
    metadata: pl.DataFrame,
    cfg: MainConfig,
    data_dir: Path,
    output_dir: Path,
    phase: str,
):
    if output_dir.exists() and not cfg.cleanup:
        print(
            f"The directory {output_dir} already exists. Skip processing spectrogram."
        )
        return

    mkdir_if_not_exists(output_dir)
    with trace("process spectrogram"):
        spectrogram_ids = (
            metadata["spectrogram_id"].unique(maintain_order=True).to_numpy()
        )
        process_fn = partial(
            process_single_spectrogram,
            data_dir=data_dir,
            phase=phase,
            output_dir=output_dir,
            dry_run=cfg.dry_run,
        )
        with get_context("spawn").Pool(cfg.env.num_workers) as pool:
            print(f"Start processing spectrogram with {cfg.env.num_workers} workers.")
            list(
                tqdm(
                    pool.imap(process_fn, spectrogram_ids),
                    total=spectrogram_ids.shape[0],
                )
            )


@hydra.main(config_path="conf", config_name="main", version_base="1.2")
def main(cfg: MainConfig):
    phase = cfg.phase if cfg.phase != "develop" else "train"

    data_dir = Path(cfg.env.data_dir)

    metadata = pl.read_csv(data_dir / f"{phase}.csv")
    if cfg.phase == "develop":
        columns_org = metadata.columns
        metadata = process_label(metadata)
        metadata = select_develop_samples(metadata)
        metadata = metadata.select(columns_org)

    output_dir_eeg = Path("eeg")
    output_dir_spectrogram = Path("spectrogram")

    preprocess_eeg(metadata, cfg, data_dir, output_dir_eeg, phase)
    preprocess_spectrogram(metadata, cfg, data_dir, output_dir_spectrogram, phase)


if __name__ == "__main__":
    main()
