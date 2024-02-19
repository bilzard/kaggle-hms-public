import torch
import torch.nn as nn
from hydra.utils import instantiate
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import get_cosine_schedule_with_warmup

from src.callback.base import Callback
from src.config import TrainerConfig


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.current: float = 0.0
        self.mean: float = 0.0
        self.sum: float = 0.0
        self.count: float = 0

    def update(self, val: float, count: float):
        self.current = val
        self.sum += val * count
        self.count += count
        self.mean = self.sum / self.count


class Trainer:
    def __init__(
        self,
        cfg: TrainerConfig,
        model: nn.Module,
        device,
        train_loader: DataLoader,
        valid_loader: DataLoader,
        epochs: int,
        callbacks: list[Callback] = [],
        mixed_precision=True,
    ):
        self.cfg = cfg
        self.model = model
        self.criterion = nn.KLDivLoss(reduction="none")
        self.device = device
        self.epochs = epochs
        self.mixed_precision = mixed_precision
        self.scaler = torch.cuda.amp.GradScaler() if self.mixed_precision else None
        self.callbacks = callbacks

        self.target_key = cfg.data.target_key
        self.pred_key = cfg.data.pred_key
        self.weight_key = cfg.data.weight_key
        self.input_keys = cfg.data.input_keys

        self.train_loader = train_loader
        self.valid_loader = valid_loader

        self._train_loss_meter = AverageMeter()
        self._valid_loss_meter = AverageMeter()

        self.configure_optimizers()

    def configure_optimizers(self):
        cfg = self.cfg
        max_steps = len(self.train_loader) * self.epochs
        self.optimizer = instantiate(
            self.cfg.optimizer,
            params=self.model.parameters(),
            lr=cfg.lr * (cfg.batch_size * cfg.num_samples_per_eeg) / 32.0,
        )
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_training_steps=max_steps,
            num_warmup_steps=int(max_steps * cfg.scheduler.warmup_ratio),
            num_cycles=0.5,
        )
        print(f"Optimizer: {self.optimizer}")

    def fit(self):
        for epoch in range(self.epochs):
            self._train_loss_meter.reset()
            self.train_epoch()
            for callback in self.callbacks:
                callback.on_train_epoch_end(self, epoch, self._train_loss_meter.mean)

            self._valid_loss_meter.reset()
            self.valid_epoch()
            for callback in self.callbacks:
                callback.on_valid_epoch_end(
                    self,
                    epoch,
                    self._valid_loss_meter.mean,
                )

    def _move_device(self, x: dict[str, torch.Tensor]):
        for k, v in x.items():
            if k in self.input_keys + [self.target_key, self.weight_key]:
                x[k] = v.to(self.device)

    def _calc_loss_with_weight(
        self, pred: torch.Tensor, target: torch.Tensor, weight: torch.Tensor
    ) -> tuple[torch.Tensor, float]:
        """
        pred: (B, C)
        target: (B, C)
        weight: (B, 1)
        """
        pred = torch.log_softmax(pred, dim=1)
        weight_sum = weight.sum().item()
        loss = self.criterion(pred, target).sum(dim=1) * weight.squeeze(1)
        loss = loss.sum() / weight_sum
        return loss, weight_sum

    def _calc_loss(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> tuple[torch.Tensor, float]:
        """
        pred: (B, C)
        target: (B, C)
        """
        pred = torch.log_softmax(pred, dim=1)
        batch_size = pred.shape[0]
        loss = self.criterion(pred, target).sum(dim=1)
        loss = loss.sum() / batch_size
        return loss, batch_size

    def train_epoch(self):
        self.model.train()
        with tqdm(self.train_loader, unit="step") as pbar:
            for batch in pbar:
                self.optimizer.zero_grad()
                self._move_device(batch)
                with torch.autocast(device_type="cuda", enabled=self.mixed_precision):
                    output = self.model(batch)
                    loss, weight_sum = self._calc_loss_with_weight(
                        output[self.pred_key],
                        batch[self.target_key],
                        batch[self.weight_key],
                    )
                    self._train_loss_meter.update(loss.item(), weight_sum)

                if self.scaler is not None:
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    self.optimizer.step()
                self.scheduler.step()

                for callback in self.callbacks:
                    callback.on_train_step_end(
                        self, batch, output, self._train_loss_meter.mean
                    )
                pbar.set_postfix({"loss": self._train_loss_meter.mean})

    def valid_epoch(self):
        self.model.eval()
        with torch.no_grad():
            with tqdm(self.valid_loader, unit="step") as pbar:
                for batch in pbar:
                    self._move_device(batch)
                    output = self.model(batch)
                    loss, weight_sum = self._calc_loss_with_weight(
                        output[self.pred_key],
                        batch[self.target_key],
                        batch[self.weight_key],
                    )
                    self._valid_loss_meter.update(loss.item(), weight_sum)

                    for callback in self.callbacks:
                        callback.on_valid_step_end(
                            self, batch, output, self._valid_loss_meter.mean
                        )
                    pbar.set_postfix({"loss": self._valid_loss_meter.mean})
