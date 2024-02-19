import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.constant import LABELS


class EvaluatorV2:
    """
    EEGごとの予測値がすでに集約されているバージョン
    """

    def __init__(
        self,
        aggregation_fn: str = "max",
    ):
        self._val_eeg_ids = []
        self._val_logits = []
        self._val_loss_per_label = np.zeros(len(LABELS), dtype=np.float32)
        self._val_weight = 0.0

        self.criterion = nn.KLDivLoss(reduction="none")
        self.pred_key = "pred"
        self.target_key = "label"
        self.weight_key = "weight"
        self.aggregation_fn = aggregation_fn

        self.clear_state()

    @torch.no_grad()
    def clear_state(self):
        self._val_eeg_ids.clear()
        self._val_logits.clear()
        self._val_loss_per_label[:] = 0.0
        self._val_weight = 0.0

    @torch.no_grad()
    def evaluate(
        self, model: nn.Module, valid_loader: DataLoader, device: str = "cuda"
    ) -> tuple[float, dict[str, float], np.ndarray, np.ndarray]:
        model.eval()
        model.to(device=device)

        with tqdm(valid_loader, unit="step") as pbar:
            for batch in pbar:
                batch["eeg"] = batch["eeg"].to(device=device)
                batch["cqf"] = batch["cqf"].to(device=device)
                output = model(batch)
                self.process_batch(batch, output)

        return self.aggregate()

    @torch.no_grad()
    def process_batch(
        self,
        batch: dict[str, torch.Tensor],
        output: dict[str, torch.Tensor],
    ):
        logit = output[self.pred_key].squeeze(0)
        label = batch[self.target_key].squeeze(0)
        weight = batch[self.weight_key].squeeze(0)
        eeg_id = batch["eeg_id"].item()

        loss = (
            (self.criterion(torch.log_softmax(logit, dim=0), label) * weight)
            .detach()
            .cpu()
            .numpy()
        )
        assert (
            loss.sum().item() >= 0
        ), f"loss: {loss}, logit: {logit.detach().cpu().numpy()}, label: {label.detach().cpu().numpy()}, weight: {weight.detach().cpu().numpy()}"
        self._val_loss_per_label += loss
        self._val_weight += weight.item()
        self._val_eeg_ids.extend(batch["eeg_id"].detach().cpu().numpy().tolist())
        self._val_logits.extend(logit.detach().cpu().numpy().tolist())
        self._val_eeg_ids.append(eeg_id)

    @torch.no_grad()
    def aggregate(self) -> tuple[float, dict[str, float], np.ndarray, np.ndarray]:
        self._val_loss_per_label /= self._val_weight
        val_loss = self._val_loss_per_label.sum()

        # eegごとの予測値をDataFrameに変換
        val_loss_per_label = dict(zip(LABELS, self._val_loss_per_label.tolist()))
        eeg_ids = np.array(self._val_eeg_ids)
        logits = np.array(self._val_logits)

        self.clear_state()

        return val_loss, val_loss_per_label, eeg_ids, logits
