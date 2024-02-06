import torch

from src.callback.base import Callback


class MetricsLogger(Callback):
    @torch.no_grad()
    def on_train_epoch_end(self, trainer, epoch: int, loss: float):
        print(f"[epoch {epoch}] train_loss: {loss:.4f}")

    @torch.no_grad()
    def on_train_step_end(
        self,
        trainer,
        batch: dict[str, torch.Tensor],
        output: dict[str, torch.Tensor],
        loss: float,
    ):
        pass

    @torch.no_grad()
    def on_valid_epoch_end(self, trainer, epoch: int, loss: float):
        print(f"[epoch {epoch}] validation_loss: {loss:.4f}")

    @torch.no_grad()
    def on_valid_step_end(
        self,
        trainer,
        batch: dict[str, torch.Tensor],
        output: dict[str, torch.Tensor],
        loss: float,
    ):
        pass
