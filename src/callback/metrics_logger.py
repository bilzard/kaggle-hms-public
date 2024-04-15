from typing import Any

import torch
import torch.nn as nn
import wandb

from src.callback.base import Callback
from src.evaluator import Evaluator


def _add_scheduler_value_to_wandb(
    data: dict[str, Any], trainer: nn.Module, name: str, group: str = "step"
):
    if hasattr(trainer, f"{name}_scheduler"):
        data[f"{group}/{name}"] = getattr(trainer, f"{name}_scheduler").value


def _add_loss_meter_value_to_wandb(
    data: dict[str, Any], trainer: nn.Module, name: str, group: str = "epoch"
):
    if hasattr(trainer, f"_train_loss_meter_{name}"):
        data[f"{group}/train_loss_{name}"] = getattr(
            trainer, f"_train_loss_meter_{name}"
        ).mean


class MetricsLogger(Callback):
    def __init__(
        self,
        aggregation_fn: str = "max",
        weight_exponent: float = 1.0,
        min_weight: float = 1e-3,
    ):
        assert aggregation_fn in [
            "max",
            "mean",
        ], f"Invalid aggregation_fn: {aggregation_fn}"

        self.aggregation_fn = aggregation_fn
        self.evaluator = Evaluator(
            aggregation_fn=aggregation_fn,
            weight_exponent=weight_exponent,
            min_weight=min_weight,
        )

        self.wandb_enabled = wandb.run is not None

    @torch.no_grad()
    def on_train_epoch_end(self, trainer, epoch: int, loss: float):
        print(f"[epoch {epoch}] train_loss: {loss:.4f}")
        if self.wandb_enabled:
            data = {
                "epoch/train_loss": loss,
                **{
                    f"epoch/lr{i}": lr
                    for i, lr in enumerate(trainer.scheduler.get_last_lr())
                },
                "epoch": epoch,
            }

            _add_scheduler_value_to_wandb(data, trainer, "forget_rate", group="epoch")
            _add_scheduler_value_to_wandb(
                data, trainer, "weight_exponent", group="epoch"
            )
            _add_scheduler_value_to_wandb(data, trainer, "min_weight", group="epoch")
            _add_scheduler_value_to_wandb(data, trainer, "max_weight", group="epoch")
            _add_scheduler_value_to_wandb(
                data, trainer, "contrastive_weight", group="epoch"
            )

            _add_loss_meter_value_to_wandb(data, trainer, "eeg", group="epoch")
            _add_loss_meter_value_to_wandb(data, trainer, "spec", group="epoch")
            _add_loss_meter_value_to_wandb(data, trainer, "contrastive", group="epoch")
            _add_loss_meter_value_to_wandb(data, trainer, "aux", group="epoch")

            wandb.log(data)

    @torch.no_grad()
    def on_train_step_end(
        self,
        trainer: nn.Module,
        batch: dict[str, torch.Tensor],
        output: dict[str, torch.Tensor],
        loss: float,
    ):
        pass

    @torch.no_grad()
    def on_valid_epoch_end(self, trainer, epoch: int, loss: float):
        if trainer.no_eval:
            return

        output = self.evaluator.aggregate()
        val_loss, val_loss_per_label = output["val_loss"], output["val_loss_per_label"]

        print(
            f"[epoch {epoch}] val_loss: {val_loss:.4f}, val_loss_per_chunk: {loss:.4f}, val_loss_per_label: ("
            + " ".join([f"{k}={v:.4f}" for k, v in val_loss_per_label.items()])
            + ")"
        )
        if self.wandb_enabled:
            data = {
                "epoch/val_loss": val_loss,
                "epoch/val_loss_per_chunk": loss,
                "epoch/table_val_loss_per_label": wandb.Table(
                    columns=["label", "loss"],
                    data=list(
                        zip(
                            list(val_loss_per_label.keys()),
                            list(val_loss_per_label.values()),
                        ),
                    ),
                ),
                **{
                    f"epoch/val_loss_{label}": loss
                    for label, loss in val_loss_per_label.items()
                },
                "epoch": epoch,
            }

            wandb.log(data)

    @torch.no_grad()
    def on_valid_step_end(
        self,
        trainer,
        batch: dict[str, torch.Tensor],
        output: dict[str, torch.Tensor],
        loss: float,
    ):
        if trainer.no_eval:
            return
        self.evaluator.process_batch(batch, output)
