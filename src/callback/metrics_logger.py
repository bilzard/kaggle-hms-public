from collections import defaultdict

import torch
import wandb
from tqdm.auto import tqdm

from src.callback.base import Callback


class MetricsLogger(Callback):
    def __init__(self, aggregation_fn: str = "max"):
        assert aggregation_fn in [
            "max",
            "mean",
        ], f"Invalid aggregation_fn: {aggregation_fn}"
        self.aggregation_fn = aggregation_fn

        self.wandb_enabled = wandb.run is not None
        self._valid_preds = defaultdict(list)
        self._valid_label = dict()
        self._valid_weight = dict()

    @torch.no_grad()
    def on_train_epoch_end(self, trainer, epoch: int, loss: float):
        print(f"[epoch {epoch}] train_loss: {loss:.4f}")
        if self.wandb_enabled:
            wandb.log({"epoch/train_loss": loss, "epoch": epoch})

    @torch.no_grad()
    def on_train_step_end(
        self,
        trainer,
        batch: dict[str, torch.Tensor],
        output: dict[str, torch.Tensor],
        loss: float,
    ):
        if self.wandb_enabled:
            wandb.log(
                {
                    f"step/lr{i}": lr
                    for i, lr in enumerate(trainer.scheduler.get_last_lr())
                }
            )

    @torch.no_grad()
    def on_valid_epoch_end(self, trainer, epoch: int, loss: float):
        val_loss = 0.0
        val_count = 0.0

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

            val_loss += (
                trainer.criterion(torch.log_softmax(pred, dim=0), label).sum() * weight
            )
            val_count += weight

        val_loss /= val_count
        self._valid_preds.clear()

        print(
            f"[epoch {epoch}] val_loss: {val_loss:.4f}, val_loss_per_chunk: {loss:.4f}"
        )
        if self.wandb_enabled:
            wandb.log(
                {
                    "epoch/val_loss": val_loss,
                    "epoch/val_loss_per_chunk": loss,
                    "epoch": epoch,
                }
            )

    @torch.no_grad()
    def on_valid_step_end(
        self,
        trainer,
        batch: dict[str, torch.Tensor],
        output: dict[str, torch.Tensor],
        loss: float,
    ):
        pred = output[trainer.pred_key]
        label = batch[trainer.target_key]
        weight = batch[trainer.weight_key]
        eeg_ids = batch["eeg_id"].detach().cpu().numpy().tolist()

        for i, eeg_id in enumerate(eeg_ids):
            self._valid_preds[eeg_id].append(pred[i].detach().cpu())
            self._valid_label[eeg_id] = label[i].detach().cpu()
            self._valid_weight[eeg_id] = weight[i].detach().cpu()
