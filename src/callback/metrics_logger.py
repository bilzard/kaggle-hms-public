import torch
import wandb

from src.callback.base import Callback
from src.evaluator import Evaluator


class MetricsLogger(Callback):
    def __init__(self, aggregation_fn: str = "max"):
        assert aggregation_fn in [
            "max",
            "mean",
        ], f"Invalid aggregation_fn: {aggregation_fn}"

        self.aggregation_fn = aggregation_fn
        self.evaluator = Evaluator(aggregation_fn=aggregation_fn)

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
            wandb.log(
                {
                    "epoch/val_loss": val_loss,
                    "epoch/val_loss_per_chunk": loss,
                    "epoch/val_loss_per_label": wandb.Table(
                        columns=list(val_loss_per_label.keys()),
                        data=[
                            list(val_loss_per_label.values()),
                        ],
                    ),
                    **{
                        f"epoch/val_loss_{label}": loss
                        for label, loss in val_loss_per_label.items()
                    },
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
        self.evaluator.process_batch(batch, output)
