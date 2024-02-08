from pathlib import Path

import torch

from src.callback.base import Callback


class SaveModelCheckpoint(Callback):
    def __init__(
        self, save_last: bool = True, save_best: bool = True, save_dir: str = "model"
    ):
        self.save_last = save_last
        self.save_dir = Path(save_dir)
        self.save_best = save_best

        self.best_loss = float("inf")

        if not self.save_dir.exists():
            self.save_dir.mkdir(parents=True, exist_ok=True)

    @torch.no_grad()
    def on_train_epoch_end(self, trainer, epoch: int, loss: float):
        pass

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
        if self.save_last:
            torch.save(
                dict(checkpoint=trainer.model.state_dict(), epoch=epoch, loss=loss),
                "model/last_model.pth",
            )
            print("*** Saved last model checkpoint ***")
        if self.save_best and loss < self.best_loss:
            torch.save(
                dict(checkpoint=trainer.model.state_dict(), epoch=epoch, loss=loss),
                "model/best_model.pth",
            )
            print("*** Saved best model checkpoint ***")
            self.best_loss = loss

    @torch.no_grad()
    def on_valid_step_end(
        self,
        trainer,
        batch: dict[str, torch.Tensor],
        output: dict[str, torch.Tensor],
        loss: float,
    ):
        pass
