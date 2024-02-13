from collections import defaultdict

import numpy as np
import polars as pl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.constant import LABELS


class Evaluator:
    def __init__(
        self,
        aggregation_fn: str = "max",
    ):
        self._valid_preds = defaultdict(list)
        self._valid_label = dict()
        self._valid_weight = dict()

        self.criterion = nn.KLDivLoss(reduction="none")
        self.pred_key = "pred"
        self.target_key = "label"
        self.weight_key = "weight"
        self.aggregation_fn = aggregation_fn

    @torch.no_grad()
    def evaluate(
        self, model: nn.Module, valid_loader: DataLoader, device: str = "cuda"
    ) -> tuple[float, dict[str, float], pl.DataFrame]:
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
        pred = output[self.pred_key]
        label = batch[self.target_key]
        weight = batch[self.weight_key]
        eeg_ids = batch["eeg_id"].detach().cpu().numpy().tolist()

        for i, eeg_id in enumerate(eeg_ids):
            self._valid_preds[eeg_id].append(pred[i].detach().cpu())
            self._valid_label[eeg_id] = label[i].detach().cpu()
            self._valid_weight[eeg_id] = weight[i].detach().cpu()

    @torch.no_grad()
    def aggregate(self) -> tuple[float, dict[str, float], pl.DataFrame]:
        val_loss_per_label = np.zeros(len(LABELS), dtype=np.float32)
        val_count = 0.0

        preds_per_eeg = []
        eeg_ids = []

        for eeg_id, preds in tqdm(self._valid_preds.items()):
            preds = torch.stack(preds, dim=0)  # (B, C)
            label = self._valid_label[eeg_id]  # (C)
            weight = self._valid_weight[eeg_id].item()

            if self.aggregation_fn == "max":
                pred = torch.max(preds, dim=0)[0]
            elif self.aggregation_fn == "mean":
                pred = preds.mean(dim=0)
            else:
                raise ValueError(f"Invalid aggregation_fn: {self.aggregation_fn}")

            # lossの計算
            val_loss_per_label += (
                (self.criterion(torch.log_softmax(pred, dim=0), label) * weight)
                .detach()
                .cpu()
                .numpy()
            )
            val_count += weight

            # eegごとの予測値
            eeg_ids.append(eeg_id)
            preds_per_eeg.append(pred.detach().cpu().numpy())  # (C)

        # lossの正規化
        val_loss_per_label /= val_count
        val_loss = val_loss_per_label.sum()

        self._valid_preds.clear()

        # eegごとの予測値をDataFrameに変換
        preds_per_eeg = np.stack(preds_per_eeg, axis=0)
        eeg_ids = np.array(eeg_ids)
        pred_df = pl.DataFrame(
            {
                "eeg_id": eeg_ids,
                **{
                    f"{label}_vote": preds_per_eeg[:, i]
                    for i, label in enumerate(LABELS)
                },
            }
        )
        val_loss_per_label = dict(zip(LABELS, val_loss_per_label.tolist()))

        return val_loss, val_loss_per_label, pred_df
