import numpy as np
from torch.utils.data import Sampler


def get_current_sampling_rate(
    epoch: int,
    saturated_epochs: int,
    initial_sampling_rate: float,
    final_sampling_rate: float,
    eps: float = 1e-6,
):
    current_sampling_rate = initial_sampling_rate - (
        initial_sampling_rate - final_sampling_rate
    ) * (epoch / (saturated_epochs + eps))

    current_sampling_rate = max(current_sampling_rate, final_sampling_rate)
    return current_sampling_rate


class LossBasedSampler(Sampler):
    """
    1 epochごとにLossの値が小さいデータを一定の割合でサンプリングする
    """

    def __init__(
        self,
        losses: np.ndarray,
        saturated_epochs: int,
        initial_sampling_rate=1.0,
        final_sampling_rate=0.75,
        shuffle=True,
        eps=1e-6,
    ):
        assert saturated_epochs >= 0
        assert 1 >= initial_sampling_rate >= final_sampling_rate >= 0

        self.epoch = 0
        self.losses = losses
        self.saturated_epochs = max(saturated_epochs, 0)
        self.initial_sampling_rate = initial_sampling_rate
        self.final_sampling_rate = final_sampling_rate
        self.shuffle = shuffle
        self.sorted_indices = np.argsort(losses)
        self.eps = eps

        self._num_samples = len(losses)

    @property
    def num_samples(self):
        return self._num_samples

    def __iter__(self):
        current_epoch = getattr(self, "epoch", 0)
        current_sampling_rate = get_current_sampling_rate(
            current_epoch,
            self.saturated_epochs,
            self.initial_sampling_rate,
            self.final_sampling_rate,
            self.eps,
        )
        self._num_samples = int(len(self.losses) * current_sampling_rate)

        selected_indices = self.sorted_indices[: self._num_samples]
        if self.shuffle:
            selected_indices = np.random.permutation(selected_indices)

        yield from selected_indices.tolist()

    def __len__(self) -> int:
        return self._num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch
