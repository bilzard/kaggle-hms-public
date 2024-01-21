import numpy as np
import polars as pl
import torch
import torchaudio.functional as AF
from einops import rearrange

from src.array_util import pad_multiple_of
from src.constant import EEG_PROBES, LABELS, PROBES


def load_eeg(eeg_id, data_dir, phase="train"):
    return pl.read_parquet(data_dir / f"{phase}_eegs/{eeg_id}.parquet")


def load_spectrogram(eeg_id, data_dir, phase="train"):
    return pl.read_parquet(data_dir / f"{phase}_spectrograms/{eeg_id}.parquet")


def process_spectrogram(spectrogram: pl.DataFrame, eps=1e-4) -> np.ndarray:
    x = 10 * np.log10(spectrogram.fill_null(0) + eps) - 30
    return x


def process_eeg(eeg: pl.DataFrame, down_sampling_rate=5) -> np.ndarray:
    eeg = eeg.select(PROBES).interpolate().fill_null(0)
    x = eeg.to_numpy()
    x = pad_multiple_of(x, down_sampling_rate)
    x = rearrange(x, "(n k) c -> n k c", k=down_sampling_rate)
    x = np.mean(x, axis=1)

    return x


def process_mask(eeg: pl.DataFrame, down_sampling_rate=5) -> np.ndarray:
    mask = eeg.select(f"mask-{probe}" for probe in EEG_PROBES).with_columns(
        pl.col(f"mask-{probe}").cast(pl.UInt8) for probe in EEG_PROBES
    )
    mask = mask[down_sampling_rate // 2 :: down_sampling_rate, :]
    return mask.to_numpy()


def quality_estimator(x: np.ndarray, T: float = 30) -> np.ndarray:
    assert T > 0.0
    return np.log1p(x.clip(0, T)) / np.log1p(T)


def process_label(metadata: pl.DataFrame, T: float = 30):
    labels = metadata.select(f"{label}_vote" for label in LABELS).to_numpy()
    total_votes = labels.sum(axis=1, keepdims=True)
    labels = labels / total_votes
    quality = quality_estimator(total_votes, T)
    return labels, quality, total_votes


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
