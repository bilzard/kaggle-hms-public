import torch


class Callback:
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
        pass

    @torch.no_grad()
    def on_valid_step_end(
        self,
        trainer,
        batch: dict[str, torch.Tensor],
        output: dict[str, torch.Tensor],
        loss: float,
    ):
        pass
