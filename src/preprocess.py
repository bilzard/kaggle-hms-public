import numpy as np
import polars as pl

from src.constant import EEG_PROBES


def load_eeg(eeg_id, data_dir, phase="train"):
    return pl.read_parquet(data_dir / f"{phase}_eegs/{eeg_id}.parquet")


def load_spectrogram(eeg_id, data_dir, phase="train"):
    return pl.read_parquet(data_dir / f"{phase}_spectrograms/{eeg_id}.parquet")


def process_spectrogram(spectrogram: pl.DataFrame, eps=1e-4) -> np.ndarray:
    x = 10 * np.log10(spectrogram.fill_null(0) + eps) - 30
    return x


def process_eeg(
    eeg: pl.DataFrame, rolling_frames=5, ekg_rolling_frames=5
) -> np.ndarray:
    eeg = (
        eeg.select(EEG_PROBES + ["EKG"])
        .interpolate()
        .with_columns(
            *[
                pl.col(probe)
                .rolling_mean(rolling_frames, min_periods=1, center=True)
                .fill_null(0)
                .alias(probe)
                for probe in EEG_PROBES
            ],
            pl.col("EKG")
            .rolling_mean(ekg_rolling_frames, min_periods=1, center=True)
            .fill_null(0)
            .alias("EKG"),
        )
    )
    return eeg.to_numpy()


def process_mask(eeg: pl.DataFrame) -> np.ndarray:
    mask = eeg.select(f"mask-{probe}" for probe in EEG_PROBES).with_columns(
        pl.col(f"mask-{probe}").cast(pl.UInt8) for probe in EEG_PROBES
    )
    return mask.to_numpy()
