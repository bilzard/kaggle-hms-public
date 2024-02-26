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
        agg_policy: str = "per_eeg_weighted",
    ):
        assert agg_policy in [
            "per_eeg_weighted",
            "per_eeg_mean",
            "per_label_weighted",
            "per_label_mean",
        ]
        self._valid_logits = defaultdict(list)
        self._valid_label = defaultdict(list)
        self._valid_weight = defaultdict(list)

        self.criterion = nn.KLDivLoss(reduction="none")
        self.pred_key = "pred"
        self.target_key = "label"
        self.weight_key = "weight"
        self.aggregation_fn = aggregation_fn
        self.input_keys = input_keys
        self.device = device
        self.agg_policy = agg_policy

    def __repr__(self):
        return f"{self.__class__.__name__}(device={self.device}, agg_policy={self.agg_policy}, aggregation_fn={self.aggregation_fn}, input_keys={self.input_keys}, pred_key={self.pred_key}, target_key={self.target_key}, weight_key={self.weight_key})"

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
            self._valid_label[eeg_id].append(label[i].detach().cpu())
            self._valid_weight[eeg_id].append(weight[i].detach().cpu())

    @torch.no_grad()
    def aggregate(self) -> tuple[float, dict[str, float], np.ndarray, np.ndarray]:
        val_loss_per_label = np.zeros(len(LABELS), dtype=np.float32)
        val_count = 0.0

        logits_per_eeg = []
        eeg_ids = []

        for eeg_id, logits in tqdm(self._valid_logits.items()):
            logits = torch.stack(logits, dim=0)  # (B, C)
            labels = torch.stack(self._valid_label[eeg_id], dim=0)  # (B, C)
            weights = torch.stack(self._valid_weight[eeg_id], dim=0)  # (B, 1)

            if self.aggregation_fn == "max":
                logit = torch.max(logits, dim=0)[0]
            elif self.aggregation_fn == "mean":
                logit = logits.mean(dim=0)
            else:
                raise ValueError(f"Invalid aggregation_fn: {self.aggregation_fn}")

            match self.agg_policy:
                case "per_eeg_weighted":
                    label = (labels * weights).sum(dim=0)  # (C)
                    label = label / label.sum()
                    pred = torch.log_softmax(logit, dim=0)  # (C)
                    loss = self.criterion(pred, label) * weights.sum()
                    val_loss_per_label += loss.detach().cpu().numpy()
                    val_count += weights.sum().item()
                case "per_eeg_mean":
                    label = labels.mean(dim=0)  # (C)
                    label = label / label.sum()
                    pred = torch.log_softmax(logit, dim=0)  # (C)
                    loss = self.criterion(pred, label)
                    val_loss_per_label += loss.detach().cpu().numpy()
                    val_count += 1
                case "per_label_weighted":
                    preds = torch.log_softmax(logits, dim=1)
                    loss = self.criterion(preds, labels)
                    loss = (loss * weights).sum(dim=0)
                    val_loss_per_label += loss.detach().cpu().numpy()
                    val_count += weights.sum().item()
                case "per_label_mean":
                    preds = torch.log_softmax(logits, dim=1)
                    loss = self.criterion(preds, labels)
                    loss = loss.mean(dim=0)
                    val_loss_per_label += loss.detach().cpu().numpy()
                    val_count += 1
                case _:
                    raise ValueError(f"Invalid agg_policy: {self.agg_policy}")

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
