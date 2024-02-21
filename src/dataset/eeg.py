import numpy as np
import polars as pl
from torch.utils.data import DataLoader, Dataset

from src.array_util import pad_multiple_of
from src.constant import LABELS
from src.transform import BaseTransform


def row_to_dict(row: dict, exclude_keys: list[str]):
    return {key: value for key, value in row.items() if key not in exclude_keys}


def pad_eeg(
    eeg: np.ndarray, cqf: np.ndarray, duration: int, padding_type: str
) -> tuple[np.ndarray, np.ndarray]:
    eeg = pad_multiple_of(
        eeg,
        duration,
        0,
        padding_type=padding_type,
        mode="reflect",
    )
    cqf = pad_multiple_of(
        cqf,
        duration,
        0,
        padding_type=padding_type,
        mode="constant",
        constant_values=0,
    )
    return eeg, cqf


def sample_eeg(
    eeg: np.ndarray,
    mask: np.ndarray,
    duration: int,
    padding_type: str,
    num_samples: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    eeg: (num_frames, num_channels)
    mask: (num_frames, num_channels)

    return
    ------
    eeg: (num_samples, duration, num_channels)
    mask: (num_samples, duration, num_channels)
    """
    eeg_org, mask_org = eeg, mask
    num_frames = eeg.shape[0]

    eegs = []
    cqfs = []
    for _ in range(num_samples):
        if num_frames < duration:
            eeg, cqf = pad_eeg(eeg_org, mask_org, duration, padding_type)
        else:
            start_frame = np.random.randint(num_frames - duration + 1)
            eeg = eeg_org[start_frame : start_frame + duration]
            cqf = mask_org[start_frame : start_frame + duration]

        eegs.append(eeg.copy())
        cqfs.append(cqf.copy())

    eeg = np.stack(eegs, axis=0)  # S, T, C
    mask = np.stack(cqfs, axis=0)  # S, T, C

    return eeg, mask


class SlidingWindowPerEegDataset(Dataset):
    def __init__(
        self,
        metadata: pl.DataFrame,
        id2eeg: dict[int, np.ndarray],
        id2cqf: dict[int, np.ndarray],
        padding_type: str = "right",
        duration: int = 2048,
        stride: int = 2048,
        weight_key: str = "weight_per_eeg",
    ):
        self.weight_key = weight_key
        self.metadata = metadata.group_by("eeg_id", maintain_order=True).agg(
            *[pl.col(f"{label}_vote_per_eeg").first() for label in LABELS],
            pl.col(self.weight_key).first(),
            *[pl.col(f"{label}_prob_per_eeg").first() for label in LABELS],
            pl.col("min_eeg_label_offset_sec").first(),
            pl.col("max_eeg_label_offset_sec").first(),
        )
        self.id2eeg = id2eeg
        self.id2cqf = id2cqf
        self.duration = duration
        self.stride = stride

        self.eeg_ids = sorted(self.metadata["eeg_id"].to_list())

        self.eeg_id2metadata = {
            row["eeg_id"]: row_to_dict(row, exclude_keys=["eeg_id"])
            for row in self.metadata.to_dicts()
        }
        self.padding_type = padding_type

    def __len__(self):
        return len(self.eeg_ids)

    def __getitem__(self, idx):
        eeg_id = self.eeg_ids[idx]
        eegs, cqfs = [], []

        eeg_org = self.id2eeg[eeg_id].astype(np.float32)
        cqf_org = self.id2cqf[eeg_id].astype(np.float32)
        eeg_org, cqf_org = pad_eeg(eeg_org, cqf_org, self.duration, self.padding_type)

        num_frames = eeg_org.shape[0]
        row = self.eeg_id2metadata[eeg_id]
        for start_frame in range(0, num_frames - self.duration + 1, self.stride):
            end_frame = start_frame + self.duration
            eeg = eeg_org[start_frame:end_frame]
            cqf = cqf_org[start_frame:end_frame]
            eegs.append(eeg.copy())
            cqfs.append(cqf.copy())

        eeg = np.stack(eegs, axis=0)
        cqf = np.stack(cqfs, axis=0)

        label = np.array(
            [row[f"{label}_prob_per_eeg"] for label in LABELS], dtype=np.float32
        )
        weight = np.array([row[self.weight_key]], dtype=np.float32)
        data = dict(eeg_id=eeg_id, eeg=eeg, cqf=cqf, label=label, weight=weight)

        return data


class SlidingWindowEegDataset(Dataset):
    def __init__(
        self,
        metadata: pl.DataFrame,
        id2eeg: dict[int, np.ndarray],
        id2cqf: dict[int, np.ndarray],
        padding_type: str = "right",
        duration: int = 2048,
        stride: int = 2048,
        weight_key: str = "weight_per_eeg",
    ):
        self.weight_key = weight_key
        self.metadata = metadata.group_by("eeg_id", maintain_order=True).agg(
            *[pl.col(f"{label}_vote_per_eeg").first() for label in LABELS],
            pl.col(self.weight_key).first(),
            *[pl.col(f"{label}_prob_per_eeg").first() for label in LABELS],
            pl.col("min_eeg_label_offset_sec").first(),
            pl.col("max_eeg_label_offset_sec").first(),
        )
        self.id2eeg = id2eeg
        self.id2cqf = id2cqf
        self.duration = duration
        self.stride = stride

        self.eeg_ids = sorted(self.metadata["eeg_id"].to_list())

        self.eeg_id2metadata = {
            row["eeg_id"]: row_to_dict(row, exclude_keys=["eeg_id"])
            for row in self.metadata.to_dicts()
        }
        self.padding_type = padding_type
        self.chunks = self._generate_chunks()

    def _generate_chunks(self):
        chunks = []
        for eeg_id in self.eeg_ids:
            eeg = self.id2eeg[eeg_id]
            num_frames = eeg.shape[0]
            for start_frame in range(0, num_frames - self.duration + 1, self.stride):
                end_frame = start_frame + self.duration
                chunks.append((eeg_id, start_frame, end_frame))

        return chunks

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        eeg_id, start_frame, end_frame = self.chunks[idx]
        row = self.eeg_id2metadata[eeg_id]

        eeg = self.id2eeg[eeg_id][start_frame:end_frame].astype(np.float32)
        cqf = self.id2cqf[eeg_id][start_frame:end_frame].astype(np.float32)

        eeg, cqf = pad_eeg(eeg, cqf, self.duration, self.padding_type)

        label = np.array(
            [row[f"{label}_prob_per_eeg"] for label in LABELS], dtype=np.float32
        )
        weight = np.array([row[self.weight_key]], dtype=np.float32)
        data = dict(eeg_id=eeg_id, eeg=eeg, cqf=cqf, label=label, weight=weight)

        return data


class UniformSamplingEegDataset(Dataset):
    """
    EEGからdurationのchunkをランダムにサンプリングする。
    """

    def __init__(
        self,
        metadata: pl.DataFrame,
        id2eeg: dict[int, np.ndarray],
        id2cqf: dict[int, np.ndarray],
        padding_type: str = "right",
        duration: int = 2048,
        transform: BaseTransform | None = None,
        num_samples_per_eeg: int = 1,
        weight_key: str = "weight_per_eeg",
    ):
        self.weight_key = weight_key
        self.metadata = metadata.group_by("eeg_id", maintain_order=True).agg(
            *[pl.col(f"{label}_vote_per_eeg").first() for label in LABELS],
            pl.col(self.weight_key).first(),
            *[pl.col(f"{label}_prob_per_eeg").first() for label in LABELS],
            pl.col("min_eeg_label_offset_sec").first(),
            pl.col("max_eeg_label_offset_sec").first(),
        )
        self.id2eeg = id2eeg
        self.id2cqf = id2cqf
        self.duration = duration
        self.transform = transform
        print(f"transform: {transform}")

        self.eeg_ids = sorted(self.metadata["eeg_id"].to_list())

        self.eeg_id2metadata = {
            row["eeg_id"]: row_to_dict(row, exclude_keys=["eeg_id"])
            for row in self.metadata.to_dicts()
        }
        self.padding_type = padding_type
        self.num_samples_per_eeg = num_samples_per_eeg

    def __len__(self):
        return len(self.eeg_ids)

    def __getitem__(self, idx):
        eeg_id = self.eeg_ids[idx]
        row = self.eeg_id2metadata[eeg_id]

        eeg = self.id2eeg[eeg_id].astype(np.float32)
        cqf = self.id2cqf[eeg_id].astype(np.float32)

        eeg, cqf = sample_eeg(
            eeg, cqf, self.duration, self.padding_type, self.num_samples_per_eeg
        )

        label = np.array(
            [row[f"{label}_prob_per_eeg"] for label in LABELS], dtype=np.float32
        )
        weight = np.array([row[self.weight_key]], dtype=np.float32)

        if self.transform is not None:
            eeg, cqf = self.transform(eeg, cqf)

        data = dict(eeg_id=eeg_id, eeg=eeg, cqf=cqf, label=label, weight=weight)

        return data


class PerEegDataset(Dataset):
    """
    EEG全体を1つずつロードする。
    """

    def __init__(
        self,
        metadata: pl.DataFrame,
        id2eeg: dict[int, np.ndarray],
        id2cqf: dict[int, np.ndarray],
        duration: int = 2048,
        padding_type: str = "right",
        is_test: bool = False,
        weight_key: str = "weight_per_eeg",
    ):
        self.weight_key = weight_key
        self.metadata = metadata.group_by("eeg_id", maintain_order=True).agg(
            *[pl.col(f"{label}_vote_per_eeg").first() for label in LABELS],
            pl.col(self.weight_key).first(),
            *[pl.col(f"{label}_prob_per_eeg").first() for label in LABELS],
            pl.col("min_eeg_label_offset_sec").first(),
            pl.col("max_eeg_label_offset_sec").first(),
        )
        self.id2eeg = id2eeg
        self.id2cqf = id2cqf
        self.is_test = is_test

        self.eeg_ids = sorted(self.metadata["eeg_id"].to_list())

        self.eeg_id2metadata = {
            row["eeg_id"]: row_to_dict(row, exclude_keys=["eeg_id"])
            for row in self.metadata.to_dicts()
        }
        self.duration = duration
        self.padding_type = padding_type

    def __len__(self):
        return len(self.eeg_ids)

    def __getitem__(self, idx):
        eeg_id = self.eeg_ids[idx]
        row = self.eeg_id2metadata[eeg_id]

        eeg = self.id2eeg[eeg_id][: self.duration].astype(np.float32)
        cqf = self.id2cqf[eeg_id][: self.duration].astype(np.float32)

        eeg, cqf = pad_eeg(eeg, cqf, self.duration, self.padding_type)
        data = dict(eeg_id=eeg_id, eeg=eeg, cqf=cqf)
        if not self.is_test:
            label = np.array(
                [row[f"{label}_prob_per_eeg"] for label in LABELS], dtype=np.float32
            )
            weight = np.array([row[self.weight_key]], dtype=np.float32)
            data |= dict(label=label, weight=weight)

        return data


class PerEegSubsampleDataset(Dataset):
    """
    ラベルに相当するEEGのchunkをランダムにサンプリングする。

    num_samples_per_eeg: 1 epoch中のEEGあたりのラベルのサンプル数
    """

    def __init__(
        self,
        metadata: pl.DataFrame,
        id2eeg: dict[int, np.ndarray],
        id2cqf: dict[int, np.ndarray],
        sampling_rate: float = 40,
        duration_sec: int = 50,
        num_samples_per_eeg: int = 1,
        pad_multiple: int = 2048,
        padding_type: str = "right",
        weight_key: str = "weight",
    ):
        self.weight_key = weight_key
        self.metadata = metadata.select(
            "eeg_id",
            *[f"{label}_prob" for label in LABELS],
            self.weight_key,
            "eeg_label_offset_seconds",
        ).to_pandas()
        self.id2eeg = id2eeg
        self.id2cqf = id2cqf
        self.sampling_rate = sampling_rate
        self.duration_sec = duration_sec

        self.num_samples_per_eeg = num_samples_per_eeg
        self.eeg_ids = sorted(self.metadata["eeg_id"].unique().tolist())

        self.eeg_id2metadata = dict()
        for eeg_id, df in self.metadata.groupby("eeg_id"):
            self.eeg_id2metadata[eeg_id] = df.to_numpy()
        self.key2idx = {key: idx for idx, key in enumerate(self.metadata.columns)}
        self.pad_multiple = pad_multiple
        self.padding_type = padding_type

    def __len__(self):
        return len(self.eeg_ids) * self.num_samples_per_eeg

    def __getitem__(self, idx):
        eeg_id = self.eeg_ids[idx // self.num_samples_per_eeg]

        this_eeg = self.eeg_id2metadata[eeg_id]
        num_samples_in_this_eeg = len(this_eeg)
        sample_idx = np.random.randint(num_samples_in_this_eeg)

        row = this_eeg[sample_idx]
        eeg_label_offset_seconds = row[self.key2idx["eeg_label_offset_seconds"]]

        start_frame = int(eeg_label_offset_seconds * self.sampling_rate)
        end_frame = start_frame + int(self.duration_sec * self.sampling_rate)

        eeg = self.id2eeg[eeg_id][start_frame:end_frame].astype(np.float32)
        cqf = self.id2cqf[eeg_id][start_frame:end_frame].astype(np.float32)

        label = np.array(
            [row[self.key2idx[f"{label}_prob"]] for label in LABELS], dtype=np.float32
        )
        weight = np.array([row[self.key2idx[self.weight_key]]], dtype=np.float32)

        eeg, cqf = pad_eeg(
            eeg,
            cqf,
            self.pad_multiple,
            self.padding_type,
        )
        data = dict(eeg_id=eeg_id, eeg=eeg, cqf=cqf, label=label, weight=weight)

        return data


def get_train_loader(
    dataset: Dataset,
    batch_size: int,
    num_workers: int,
    pin_memory: bool = True,
    shuffle: bool = True,
    **kwargs,
):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
        **kwargs,
    )


def get_valid_loader(
    dataset: Dataset,
    batch_size: int,
    num_workers: int,
    pin_memory: bool = True,
    **kwargs,
):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        **kwargs,
    )
