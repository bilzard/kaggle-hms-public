from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.constant import LABELS


class Evaluator:
    def __init__(
        self,
        device: str = "cuda",
        input_keys: list[str] = ["eeg", "cqf"],
        aggregation_fn: str = "max",
    ):
        self._valid_logits = defaultdict(list)
        self._valid_label = dict()
        self._valid_weight = dict()

        self.criterion = nn.KLDivLoss(reduction="none")
        self.pred_key = "pred"
        self.target_key = "label"
        self.weight_key = "weight"
        self.aggregation_fn = aggregation_fn
        self.input_keys = input_keys
        self.device = device

    def _move_device(self, x: dict[str, torch.Tensor]):
        for k, v in x.items():
            if k in self.input_keys + [self.target_key, self.weight_key]:
                x[k] = v.to(self.device)

    @torch.no_grad()
    def evaluate(
        self, model: nn.Module, valid_loader: DataLoader, device: str = "cuda"
    ) -> tuple[float, dict[str, float], np.ndarray, np.ndarray]:
        model.eval()
        model.to(device=device)
        valid_loader.dataset.reset()  # type: ignore

        with tqdm(valid_loader, unit="step") as pbar:
            for batch in pbar:
                self._move_device(batch)
                output = model(batch)
                self.process_batch(batch, output)

        return self.aggregate()

    @torch.no_grad()
    def process_batch(
        self,
        batch: dict[str, torch.Tensor],
        output: dict[str, torch.Tensor],
    ):
        logit = output[self.pred_key]
        label = batch[self.target_key]
        weight = batch[self.weight_key]
        eeg_ids = batch["eeg_id"].detach().cpu().numpy().tolist()

        for i, eeg_id in enumerate(eeg_ids):
            self._valid_logits[eeg_id].append(logit[i].detach().cpu())
            self._valid_label[eeg_id] = label[i].detach().cpu()
            self._valid_weight[eeg_id] = weight[i].detach().cpu()

    @torch.no_grad()
    def aggregate(self) -> tuple[float, dict[str, float], np.ndarray, np.ndarray]:
        val_loss_per_label = np.zeros(len(LABELS), dtype=np.float32)
        val_count = 0.0

        logits_per_eeg = []
        eeg_ids = []

        for eeg_id, logits in tqdm(self._valid_logits.items()):
            logits = torch.stack(logits, dim=0)  # (B, C)
            label = self._valid_label[eeg_id]  # (C)
            weight = self._valid_weight[eeg_id].item()

            if self.aggregation_fn == "max":
                logit = torch.max(logits, dim=0)[0]
            elif self.aggregation_fn == "mean":
                logit = logits.mean(dim=0)
            else:
                raise ValueError(f"Invalid aggregation_fn: {self.aggregation_fn}")

            # lossの計算
            val_loss_per_label += (
                (self.criterion(torch.log_softmax(logit, dim=0), label) * weight)
                .detach()
                .cpu()
                .numpy()
            )
            val_count += weight

            # eegごとの予測値
            eeg_ids.append(eeg_id)
            logits_per_eeg.append(logit.detach().cpu().numpy())  # (C)

        # lossの正規化
        val_loss_per_label /= val_count
        val_loss = val_loss_per_label.sum()

        self._valid_logits.clear()

        # eegごとの予測値をDataFrameに変換
        logits_per_eeg = np.stack(logits_per_eeg, axis=0)
        eeg_ids = np.array(eeg_ids)
        val_loss_per_label = dict(zip(LABELS, val_loss_per_label.tolist()))

        return val_loss, val_loss_per_label, eeg_ids, logits_per_eeg
