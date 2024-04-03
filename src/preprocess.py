from itertools import product
from pathlib import Path

import numpy as np
import polars as pl
import torch
import torchaudio.functional as AF
from einops import rearrange

from src.constant import EEG_PROBES, LABELS, PROBE2IDX, PROBES


def load_eeg(eeg_id: int, data_dir: Path, phase: str):
    return pl.read_parquet(data_dir / f"{phase}_eegs/{eeg_id}.parquet")


def load_spectrogram(eeg_id: int, data_dir: Path, phase: str):
    return pl.read_parquet(data_dir / f"{phase}_spectrograms/{eeg_id}.parquet")


def map_log_scale(spectrogram: np.ndarray, eps=1e-4) -> np.ndarray:
    x = 10 * np.log10(spectrogram + eps) - 30
    return x


def drop_leftmost_nulls_in_array(x: np.ndarray) -> tuple[np.ndarray, int]:
    is_non_null = ~np.isnan(x).any(axis=1)
    first_non_null_idx = np.argmax(is_non_null).item()
    x = x[first_non_null_idx:]
    return x, first_non_null_idx


def drop_rightmost_nulls_in_array(x: np.ndarray) -> tuple[np.ndarray, int]:
    x, first_non_null_idx = drop_leftmost_nulls_in_array(x[::-1])
    return x[::-1], first_non_null_idx


def process_eeg(
    eeg_df: pl.DataFrame,
    down_sampling_rate=5,
    minimum_seq_length=2000,
    clip_val: float = 65504,  # np.finfo(np.float16).max
    fill_nan_with: float = 0.0,
    apply_filter: bool = False,
    cutoff_freqs: tuple[float | None, float | None] = (None, None),
    reject_freq: float | None = None,
    device: str = "cpu",
    sampling_rate: int = 200,
    drop_leftmost_nulls: bool = False,
    pad_mode: str = "constant",
) -> tuple[np.ndarray, np.ndarray]:
    eeg_df = eeg_df.select(PROBES)
    eeg_df = eeg_df.interpolate()
    x = eeg_df.to_numpy()

    if apply_filter:
        x = rearrange(x, "t c -> c t")
        x = do_apply_filter(
            x,
            sampling_rate=sampling_rate,
            cutoff_freqs=cutoff_freqs,
            reject_freq=reject_freq,
            device=device,
        )
        x = rearrange(x, "c t -> t c")

    # subsample
    x = rearrange(x, "(n k) c -> n k c", k=down_sampling_rate)
    x = np.nanmean(x, axis=1)
    x[:, :19] = x[:, :19] - np.nanmedian(x[:, :19])
    x = np.clip(x, -clip_val, clip_val)

    # calc pad mask
    pad_mask = ~np.isnan(x).any(axis=1)

    # drop left/rightmost nulls
    x, pad_left = drop_leftmost_nulls_in_array(x) if drop_leftmost_nulls else (x, 0)
    x, pad_right = drop_rightmost_nulls_in_array(x)
    num_frames, _ = x.shape
    pad_right = max(0, minimum_seq_length - num_frames)
    x = np.pad(x, ((pad_left, pad_right), (0, 0)), mode=pad_mode)  # type: ignore

    x = np.nan_to_num(x, nan=fill_nan_with)
    return x, pad_mask


def do_apply_filter(
    xa: np.ndarray,
    sampling_rate: int = 40,
    cutoff_freqs: tuple[float | None, float | None] = (None, None),
    reject_freq: float | None = 60,
    device="cpu",
):
    """
    xa: ch t
    x: ch t
    """
    x = torch.from_numpy(xa).float().to(device).unsqueeze(0)
    if cutoff_freqs[0] is not None:
        x = AF.highpass_biquad(x, sampling_rate, cutoff_freqs[0])
    if cutoff_freqs[1] is not None:
        x = AF.lowpass_biquad(x, sampling_rate, cutoff_freqs[1])
    if reject_freq is not None:
        x = AF.bandreject_biquad(x, sampling_rate, reject_freq)

    x = x.squeeze(0).detach().cpu().numpy()
    return x


def do_process_cqf(
    eeg_df: pl.DataFrame,
    kernel_size: int = 13,
    top_k: int = 3,
    eps: float = 1e-4,
    distance_threshold: float = 10.0,
    distance_metric: str = "l2",
    normalize_type: str = "top-k",
) -> pl.DataFrame:
    eeg_df = eeg_df.with_columns(
        # L2 distance of (p1, p2)
        pl.col(p1)
        .sub(pl.col(p2))
        .pow(2)
        .ewm_mean(half_life=kernel_size, min_periods=1)
        .add(eps)
        .alias(f"l2-dist-{p1}-{p2}")
        for p1, p2 in product(EEG_PROBES, EEG_PROBES)
    )

    idxs = []
    for p1 in EEG_PROBES:
        rx = eeg_df.select(
            f"{distance_metric}-dist-{p1}-{p2}" for p2 in EEG_PROBES
        ).to_numpy()
        top_k_indices = np.argsort(rx, axis=1)[:, 1 : top_k + 1]
        top_k_values = np.take_along_axis(rx, top_k_indices, axis=1)
        top_k_dist = top_k_values.mean(axis=1)
        eeg_df = eeg_df.with_columns(
            pl.Series(top_k_dist).alias(f"top-k-dist-{p1}"),
        )
        idxs.append(top_k_indices)

    xs = eeg_df.select(f"top-k-dist-{p1}" for p1 in EEG_PROBES).to_numpy()  # (N, 19)
    global_top_k_indices = np.argsort(xs, axis=1)[:, :top_k]
    global_top_k_values = np.take_along_axis(xs, global_top_k_indices, axis=1)
    global_top_k_dist = global_top_k_values.mean(axis=1)
    global_median_dist = np.median(xs, axis=1)
    idxs = np.stack(idxs, axis=1)  # (N, 19, top_k)

    for p1 in EEG_PROBES:
        local_top_k_idxs = idxs[:, PROBE2IDX[p1], :]
        local_top_k_values = np.take_along_axis(xs, local_top_k_idxs, axis=1)
        local_top_k_dist = local_top_k_values.mean(axis=1)
        eeg_df = (
            eeg_df.with_columns(
                pl.Series(local_top_k_dist).alias(f"local-top-k-dist-{p1}"),
                pl.Series(global_top_k_dist).alias(f"global-top-k-dist-{p1}"),
                pl.Series(global_median_dist).alias(f"global-median-dist-{p1}"),
            )
            .with_columns(
                pl.col(f"top-k-dist-{p1}")
                .truediv(pl.col(f"local-top-k-dist-{p1}").add(eps))
                .alias(f"LOF-{p1}"),
            )
            .with_columns(
                pl.col(f"top-k-dist-{p1}")
                .truediv(pl.col(f"global-{normalize_type}-dist-{p1}").add(eps))
                .alias(f"GOF-{p1}"),
            )
        )

    for p in EEG_PROBES:
        eeg_df = eeg_df.with_columns(
            pl.col(f"GOF-{p}").lt(distance_threshold).alias(f"mask-{p}"),
            pl.lit(1.0)
            .truediv(pl.col(f"GOF-{p}").clip(0).truediv(distance_threshold).add(1.0))
            .alias(f"CQF-{p}"),
        )

    return eeg_df


def process_spectrogram(spectrogram: pl.DataFrame, null_value=0) -> np.ndarray:
    spec = spectrogram.drop("time").fill_null(null_value).to_numpy()
    spec = rearrange(spec, "t (c f) -> f t c", c=4)
    spec = map_log_scale(spec)
    return spec


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))


def calc_weight(x: np.ndarray, T: float = 3, alpha: float = 5) -> np.ndarray:
    assert T > 0.0
    return sigmoid((x - alpha) / T)


def process_label(
    metadata: pl.DataFrame,
    population_power: float = 1.0,
    diversity_power: float = 0.0,
    max_votes: int = 28,
    add_dummy_label: bool = False,
    only_use_sp_center: bool = False,
) -> pl.DataFrame:
    if add_dummy_label:
        metadata = metadata.with_columns(
            *[pl.lit(1).alias(f"{label}_vote") for label in LABELS],
            pl.col("eeg_id").alias("label_id"),
            pl.lit(0).cast(pl.Int64).alias("eeg_label_offset_seconds"),
            pl.lit(0).cast(pl.Int64).alias("spectrogram_label_offset_seconds"),
        )

    metadata = (
        metadata.with_columns(
            (
                (
                    pl.col("seizure_vote").diff().fill_null(0).over("eeg_id").eq(0)
                    & pl.col("lpd_vote").diff().fill_null(0).over("eeg_id").eq(0)
                    & pl.col("gpd_vote").diff().fill_null(0).over("eeg_id").eq(0)
                    & pl.col("lrda_vote").diff().fill_null(0).over("eeg_id").eq(0)
                    & pl.col("grda_vote").diff().fill_null(0).over("eeg_id").eq(0)
                    & pl.col("other_vote").diff().fill_null(0).over("eeg_id").eq(0)
                ).not_()
            ).alias("_is_changed")
        )
        .with_columns(
            pl.col("_is_changed").cum_sum().over("eeg_id").alias("label_group_id"),
        )
        .with_columns(
            pl.col("label_group_id")
            .count()
            .over("eeg_id")
            .alias("num_label_groups_per_eeg"),
            pl.cum_count()
            .sub(0.5)
            .over("eeg_id", "label_group_id")
            .alias("_label_count"),
        )
        .with_columns(
            pl.col("_label_count")
            .median()
            .over("eeg_id", "label_group_id")
            .alias("_median_label_count")
        )
        .with_columns(
            pl.col("_label_count")
            .sub(pl.col("_median_label_count"))
            .abs()
            .lt(1)
            .alias("is_sp_center")
        )
        .drop("_is_changed", "_label_count", "_median_label_count")
    )
    if only_use_sp_center:
        metadata = metadata.filter(pl.col("is_sp_center"))

    total_votes = metadata.select(f"{label}_vote" for label in LABELS).fold(
        lambda s1, s2: s1 + s2
    )
    metadata = (
        metadata.with_columns(
            pl.Series(total_votes).alias("total_votes"),
        )
        .with_columns(
            pl.col(f"{label}_vote").sum().over("eeg_id").alias(f"{label}_vote_per_eeg")
            for label in LABELS
        )
        .with_columns(
            pl.col("label_id").count().over("eeg_id").alias("label_count_per_eeg")
        )
        .with_columns(
            pl.col(f"{label}_vote_per_eeg")
            .truediv(pl.col("label_count_per_eeg"))
            .alias(f"{label}_vote_per_eeg")
            for label in LABELS
        )
    )
    total_votes_per_eeg = metadata.select(
        f"{label}_vote_per_eeg" for label in LABELS
    ).fold(lambda s1, s2: s1 + s2)
    metadata = (
        metadata.with_columns(
            pl.Series(total_votes_per_eeg).alias("total_votes_per_eeg")
        )
        .with_columns(
            pl.col("total_votes").truediv(max_votes).alias("population"),
            pl.col("total_votes_per_eeg")
            .truediv(max_votes)
            .alias("population_per_eeg"),
        )
        .with_columns(
            pl.col(f"{label}_vote")
            .truediv(pl.col("total_votes"))
            .alias(f"{label}_prob")
            for label in LABELS
        )
        .with_columns(
            pl.col(f"{label}_vote_per_eeg")
            .truediv(pl.col("total_votes_per_eeg"))
            .alias(f"{label}_prob_per_eeg")
            for label in LABELS
        )
    )
    unique_vote_count = (
        metadata.select("eeg_id", *[f"{label}_vote" for label in LABELS])
        .unique()
        .group_by("eeg_id", maintain_order=True)
        .agg(pl.count().alias("num_unique_vote_combinations_per_eeg"))
    )
    metadata = metadata.join(unique_vote_count, on="eeg_id")

    #
    metadata = (
        metadata.with_columns(
            pl.col("eeg_label_offset_seconds")
            .max()
            .over("eeg_id")
            .add(50)
            .alias("duration_sec"),
        )
        .with_columns(
            pl.col("duration_sec").truediv(60).alias("duration_min"),
        )
        .with_columns(
            pl.col("label_id").count().over("eeg_id").alias("num_labels_per_eeg"),
        )
        .with_columns(
            pl.col("num_unique_vote_combinations_per_eeg")
            .truediv(pl.col("num_labels_per_eeg"))
            .alias("num_unique_vote_combinations_per_label"),
            pl.col("num_labels_per_eeg")
            .truediv(pl.col("duration_sec"))
            .alias("num_labels_per_duration_sec"),
            pl.col("num_unique_vote_combinations_per_eeg")
            .truediv(pl.col("duration_sec"))
            .alias("num_unique_vote_combinations_per_duration_sec"),
        )
        .with_columns(
            pl.col("eeg_label_offset_seconds")
            .min()
            .over("eeg_id")
            .alias("min_eeg_label_offset_sec"),
            pl.col("eeg_label_offset_seconds")
            .max()
            .over("eeg_id")
            .alias("max_eeg_label_offset_sec"),
        )
        .with_columns(
            pl.col("num_labels_per_eeg").log().alias("log_num_labels_per_eeg"),
        )
        .with_columns(
            pl.col("num_unique_vote_combinations_per_eeg")
            .truediv("num_labels_per_eeg")
            .alias("diversity")
        )
        .with_columns(
            pl.col("diversity").pow(diversity_power).alias("diversity_weight"),
            pl.col("population").pow(population_power).alias("population_weight"),
            pl.col("population_per_eeg")
            .pow(population_power)
            .alias("population_per_eeg_weight"),
        )
        .with_columns(
            pl.col("population_weight").mul(pl.col("diversity_weight")).alias("weight")
        )
        .with_columns(
            pl.col("population_per_eeg_weight")
            .mul(pl.col("diversity_weight"))
            .alias("weight_per_eeg")
        )
    )

    return metadata


def select_develop_samples(
    metadata: pl.DataFrame, num_samples: int = 2640, duration_sec: int = 50, seed=42
) -> pl.DataFrame:
    return metadata.filter(pl.col("duration_sec").eq(duration_sec)).sample(
        num_samples, with_replacement=False, seed=seed
    )
