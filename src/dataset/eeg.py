import numpy as np
import polars as pl
from torch.utils.data import DataLoader, Dataset

from src.array_util import pad_multiple_of
from src.constant import LABELS


class PerEegSubsampleDataset(Dataset):
    def __init__(
        self,
        metadata: pl.DataFrame,
        id2eeg: dict[int, np.ndarray],
        sampling_rate: float = 40,
        duration_sec: int = 50,
        num_samples_per_eeg: int = 1,
        pad_multiple: int = 512,
        padding_type: str = "both",
    ):
        self.metadata = metadata.to_pandas()
        self.id2eeg = id2eeg
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
        eeg = pad_multiple_of(eeg, self.pad_multiple, 0, padding_type=self.padding_type)
        label = np.array(
            [row[self.key2idx[f"{label}_prob"]] for label in LABELS], dtype=np.float32
        )

        return eeg, label


class PerLabelDataset(Dataset):
    def __init__(
        self,
        metadata: pl.DataFrame,
        id2eeg: dict[int, np.ndarray],
        sampling_rate: float = 40,
        duration_sec: int = 50,
        pad_multiple: int = 512,
        padding_type: str = "both",
    ):
        self.metadata = metadata.to_pandas()
        self.id2eeg = id2eeg
        self.sampling_rate = sampling_rate
        self.duration_sec = duration_sec

        self.rows = self.metadata.to_numpy()
        self.key2idx = {key: idx for idx, key in enumerate(self.metadata.columns)}
        self.pad_multiple = pad_multiple
        self.padding_type = padding_type

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.rows[idx]

        eeg_id = row[self.key2idx["eeg_id"]]
        eeg_label_offset_seconds = row[self.key2idx["eeg_label_offset_seconds"]]

        start_frame = int(eeg_label_offset_seconds * self.sampling_rate)
        end_frame = start_frame + int(self.duration_sec * self.sampling_rate)

        eeg = self.id2eeg[eeg_id][start_frame:end_frame].astype(np.float32)
        eeg = pad_multiple_of(eeg, self.pad_multiple, 0, padding_type=self.padding_type)
        label = np.array(
            [row[self.key2idx[f"{label}_prob"]] for label in LABELS], dtype=np.float32
        )

        return eeg, label


def get_train_loader(
    dataset: PerEegSubsampleDataset, batch_size: int, num_workers: int
):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )


def get_valid_loader(dataset: PerLabelDataset, batch_size: int, num_workers: int):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
