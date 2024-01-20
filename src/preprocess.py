import numpy as np
import polars as pl
import torch
import torchaudio.functional as AF

from src.constant import EEG_PROBES, PROBES


def load_eeg(eeg_id, data_dir, phase="train"):
    return pl.read_parquet(data_dir / f"{phase}_eegs/{eeg_id}.parquet")


def load_spectrogram(eeg_id, data_dir, phase="train"):
    return pl.read_parquet(data_dir / f"{phase}_spectrograms/{eeg_id}.parquet")


def process_spectrogram(spectrogram: pl.DataFrame, eps=1e-4) -> np.ndarray:
    x = 10 * np.log10(spectrogram.fill_null(0) + eps) - 30
    return x


def process_eeg(eeg: pl.DataFrame, down_sampling_rate=5) -> np.ndarray:
    eeg = (
        eeg.select(PROBES)
        .interpolate()
        .with_columns(
            *[
                pl.col(probe)
                .rolling_mean(down_sampling_rate, min_periods=1, center=True)
                .fill_null(0)
                .alias(probe)
                for probe in PROBES
            ],
        )
    )
    x = eeg.to_numpy()
    x = x[down_sampling_rate // 2 :: down_sampling_rate, :]

    return x


def process_mask(eeg: pl.DataFrame, down_sampling_rate=5) -> np.ndarray:
    mask = eeg.select(f"mask-{probe}" for probe in EEG_PROBES).with_columns(
        pl.col(f"mask-{probe}").cast(pl.UInt8) for probe in EEG_PROBES
    )
    mask = mask[down_sampling_rate // 2 :: down_sampling_rate, :]
    return mask.to_numpy()


def do_apply_filter(
    xa: np.ndarray,
    sampling_rate: int = 40,
    cutoff_freqs: tuple[float, float] = (0.5, 50),
    device="cpu",
):
    """
    x: (n_samples, )
    """
    x = torch.from_numpy(xa).float().to(device).unsqueeze(0)
    x = AF.highpass_biquad(x, sampling_rate, cutoff_freqs[0])
    x = AF.lowpass_biquad(x, sampling_rate, cutoff_freqs[1])
    x = x.squeeze(0).detach().cpu().numpy()
    return x
