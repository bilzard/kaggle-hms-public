import torch
import wandb

from src.callback.base import Callback


class MetricsLogger(Callback):
    def __init__(self):
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
        print(f"[epoch {epoch}] validation_loss: {loss:.4f}")
        if self.wandb_enabled:
            wandb.log({"epoch/val_loss": loss, "epoch": epoch})

    @torch.no_grad()
    def on_valid_step_end(
        self,
        trainer,
        batch: dict[str, torch.Tensor],
        output: dict[str, torch.Tensor],
        loss: float,
    ):
        pass
