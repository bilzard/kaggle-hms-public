import numpy as np
import polars as pl
from torch.utils.data import DataLoader, Dataset

from src.array_util import pad_multiple_of
from src.constant import LABELS


def row_to_dict(row: dict, exclude_keys: list[str]):
    return {key: value for key, value in row.items() if key not in exclude_keys}


class SlidingWindowEegDataset(Dataset):
    def __init__(
        self,
        metadata: pl.DataFrame,
        id2eeg: dict[int, np.ndarray],
        id2cqf: dict[int, np.ndarray],
        padding_type: str = "right",
        duration: int = 2048,
        stride: int = 2048,
    ):
        self.metadata = metadata.group_by("eeg_id").agg(
            *[pl.col(f"{label}_vote_per_eeg").first() for label in LABELS],
            pl.col("total_votes_per_eeg").first(),
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

        eeg = pad_multiple_of(
            eeg,
            self.duration,
            0,
            padding_type=self.padding_type,
            mode="reflect",
        )
        cqf = pad_multiple_of(
            cqf,
            self.duration,
            0,
            padding_type=self.padding_type,
            mode="constant",
            constant_values=0,
        )

        label = np.array(
            [row[f"{label}_prob_per_eeg"] for label in LABELS], dtype=np.float32
        )
        weight = np.array([row["total_votes_per_eeg"]], dtype=np.float32)
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
    ):
        self.metadata = metadata.group_by("eeg_id").agg(
            *[pl.col(f"{label}_vote_per_eeg").first() for label in LABELS],
            pl.col("total_votes_per_eeg").first(),
            *[pl.col(f"{label}_prob_per_eeg").first() for label in LABELS],
            pl.col("min_eeg_label_offset_sec").first(),
            pl.col("max_eeg_label_offset_sec").first(),
        )
        self.id2eeg = id2eeg
        self.id2cqf = id2cqf
        self.duration = duration

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
        row = self.eeg_id2metadata[eeg_id]

        eeg = self.id2eeg[eeg_id].astype(np.float32)
        cqf = self.id2cqf[eeg_id].astype(np.float32)
        num_frames = eeg.shape[0]

        if num_frames < self.duration:
            eeg = pad_multiple_of(
                eeg,
                self.duration,
                0,
                padding_type=self.padding_type,
                mode="reflect",
            )
            cqf = pad_multiple_of(
                cqf,
                self.duration,
                0,
                padding_type=self.padding_type,
                mode="constant",
                constant_values=0,
            )
        else:
            start_frame = np.random.randint(num_frames - self.duration + 1)
            eeg = eeg[start_frame : start_frame + self.duration]
            cqf = cqf[start_frame : start_frame + self.duration]

        label = np.array(
            [row[f"{label}_prob_per_eeg"] for label in LABELS], dtype=np.float32
        )
        weight = np.array([row["total_votes_per_eeg"]], dtype=np.float32)
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
        id2cqf: dict[int, np.ndarray] | None = None,
        pad_multiple: int = 2048,
        padding_type: str = "right",
    ):
        self.metadata = metadata.group_by("eeg_id").agg(
            *[pl.col(f"{label}_vote_per_eeg").first() for label in LABELS],
            pl.col("total_votes_per_eeg").first(),
            *[pl.col(f"{label}_prob_per_eeg").first() for label in LABELS],
            pl.col("min_eeg_label_offset_sec").first(),
            pl.col("max_eeg_label_offset_sec").first(),
        )
        self.id2eeg = id2eeg
        self.id2cqf = id2cqf

        self.eeg_ids = sorted(self.metadata["eeg_id"].to_list())

        self.eeg_id2metadata = {
            row["eeg_id"]: row_to_dict(row, exclude_keys=["eeg_id"])
            for row in self.metadata.to_dicts()
        }
        self.pad_multiple = pad_multiple
        self.padding_type = padding_type

    def __len__(self):
        return len(self.eeg_ids)

    def __getitem__(self, idx):
        eeg_id = self.eeg_ids[idx]
        row = self.eeg_id2metadata[eeg_id]

        eeg = self.id2eeg[eeg_id].astype(np.float32)
        eeg = pad_multiple_of(
            eeg, self.pad_multiple, 0, padding_type=self.padding_type, mode="reflect"
        )
        label = np.array(
            [row[f"{label}_prob_per_eeg"] for label in LABELS], dtype=np.float32
        )
        weight = np.array([row["total_votes_per_eeg"]], dtype=np.float32)
        data = dict(eeg_id=eeg_id, eeg=eeg, label=label, weight=weight)

        if self.id2cqf is not None:
            cqf = self.id2cqf[eeg_id].astype(np.float32)
            cqf = pad_multiple_of(
                cqf,
                self.pad_multiple,
                0,
                padding_type=self.padding_type,
                mode="constant",
                constant_values=0,
            )
            data["cqf"] = cqf

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
        id2cqf: dict[int, np.ndarray] | None = None,
        sampling_rate: float = 40,
        duration_sec: int = 50,
        num_samples_per_eeg: int = 1,
        pad_multiple: int = 2048,
        padding_type: str = "right",
    ):
        self.metadata = metadata.select(
            "eeg_id",
            *[f"{label}_prob" for label in LABELS],
            "total_votes",
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
        eeg = pad_multiple_of(
            eeg, self.pad_multiple, 0, padding_type=self.padding_type, mode="reflect"
        )
        label = np.array(
            [row[self.key2idx[f"{label}_prob"]] for label in LABELS], dtype=np.float32
        )
        weight = np.array([row[self.key2idx["total_votes"]]], dtype=np.float32)
        data = dict(eeg_id=eeg_id, eeg=eeg, label=label, weight=weight)

        if self.id2cqf is not None:
            cqf = self.id2cqf[eeg_id][start_frame:end_frame].astype(np.float32)
            cqf = pad_multiple_of(
                cqf,
                self.pad_multiple,
                0,
                padding_type=self.padding_type,
                mode="constant",
                constant_values=0,
            )
            data["cqf"] = cqf

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
