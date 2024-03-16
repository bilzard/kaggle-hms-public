import torch
import wandb

from src.callback.base import Callback
from src.evaluator import Evaluator


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
                | {
                    "step/forget_rate": trainer.forget_rate_scheduler.value,
                    "step/weight_exponent": trainer.weight_exponent_scheduler.value,
                }
            )

    @torch.no_grad()
    def on_valid_epoch_end(self, trainer, epoch: int, loss: float):
        val_loss, val_loss_per_label, _, _ = self.evaluator.aggregate()

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
                **{
                    f"epoch/lr{i}": lr
                    for i, lr in enumerate(trainer.scheduler.get_last_lr())
                },
                "epoch/forget_rate": trainer.forget_rate_scheduler.value,
                "epoch/weight_exponent": trainer.weight_exponent_scheduler.value,
                "epoch/min_weight": trainer.min_weight_scheduler.value,
                "epoch": epoch,
            }

            if (
                hasattr(trainer, "_train_loss_meter_eeg")
                and hasattr(trainer, "_train_loss_meter_spec")
                and hasattr(trainer, "_train_loss_meter_contrastive")
            ):
                data |= {
                    "epoch/train_loss_eeg": getattr(
                        trainer, "_train_loss_meter_eeg"
                    ).mean,
                    "epoch/train_loss_spec": getattr(
                        trainer, "_train_loss_meter_spec"
                    ).mean,
                    "epoch/train_loss_contrastive": getattr(
                        trainer, "_train_loss_meter_contrastive"
                    ).mean,
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
        self.evaluator.process_batch(batch, output)
