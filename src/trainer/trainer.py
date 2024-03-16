import numpy as np
import torch
import torch.nn as nn
from hydra.utils import instantiate
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import get_cosine_schedule_with_warmup

from src.callback.base import Callback
from src.config import TrainerConfig
from src.scheduler import LinearScheduler
from src.train_util import AverageMeter, get_lr_params
from src.trainer.base import BaseTrainer
from src.trainer.util import calc_weight_sum


class Trainer(BaseTrainer):
    """
    teacherモデルの予測値から算出されるlossを使ってbatch内のサンプルを間引きするtrainer。
    間引き率は学習step/epochごとにスケジュールされる。
    """

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
        teacher_model: nn.Module | None = None,
    ):
        super().__init__(cfg)
        self.model = model
        self.teacher_model = teacher_model
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
        assert len(cfg.class_weights) == 6
        self.class_weights = np.array(cfg.class_weights) ** cfg.class_weight_exponent
        self.class_weights /= self.class_weights.sum()
        self.class_weights *= 6

        print(f"** class_weights: {self.class_weights}")

        self.configure_optimizers()
        self.clear_log()
        self.log_architecture()

    def log_architecture(self):
        self.write_log("Optimizer", str(self.optimizer))
        self.write_log("Train dataset", str(self.train_loader.dataset))
        self.write_log("Valid dataset", str(self.valid_loader.dataset))
        self.write_log("Model:", str(self.model))

    def configure_optimizers(self):
        cfg = self.cfg
        max_steps = len(self.train_loader) * self.epochs
        base_lr = cfg.lr * (cfg.batch_size * cfg.num_samples_per_eeg) / 32.0
        params = get_lr_params(
            self.model,
            base_lr,
            cfg.lr_adjustments,
            no_decay_bias_params=cfg.no_decay_bias_params,
        )
        self.optimizer = instantiate(
            self.cfg.optimizer,
            params=params,
            lr=base_lr,
            weight_decay=cfg.weight_decay,
            _convert_="object",
        )
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_training_steps=max_steps,
            num_warmup_steps=int(max_steps * cfg.scheduler.warmup_ratio),
            num_cycles=0.5,
        )
        self.forget_rate_scheduler = LinearScheduler(
            initial_value=0.0,
            target_value=cfg.distillation.target_forget_rate,
            target_step=len(self.train_loader) * cfg.distillation.target_epochs,
        )
        self.weight_exponent_scheduler = LinearScheduler(
            initial_value=1.0,
            target_value=cfg.label.schedule.target_weight_exponent,
            target_step=len(self.train_loader) * cfg.label.schedule.target_epochs,
        )
        print(f"* target_step: {self.forget_rate_scheduler.target_step}")
        print(f"* target_forget_rate: {self.forget_rate_scheduler.target_value}")
        print(f"* teacher_model: {self.teacher_model is not None}")
        print(
            f"* target_weight_exponent: {self.weight_exponent_scheduler.target_value}"
        )

    def fit(self):
        for epoch in range(self.epochs):
            self._train_loss_meter.reset()

            if self.cfg.pseudo_label.enabled:
                self.train_loader.sampler.set_epoch(epoch)  # type: ignore
                print(
                    f"epoch: {epoch}, num_samples: {self.train_loader.sampler.num_samples}"  # type: ignore
                )

            self.train_epoch(epoch)
            for callback in self.callbacks:
                callback.on_train_epoch_end(self, epoch, self._train_loss_meter.mean)

            self._valid_loss_meter.reset()
            self.valid_loader.dataset.reset()  # type: ignore
            self.valid_epoch(epoch)
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

    def _calc_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        weight: torch.Tensor | None = None,
        aggregate: bool = True,
    ) -> tuple[torch.Tensor, float]:
        """
        pred: b k c
        target: b k c
        weight: b k
        """
        assert (
            pred.ndim == 3 and target.ndim == 3
        ), f"Invalid shape: {pred.shape}, {target.shape}"
        assert weight is None or weight.ndim == 2, f"Invalid shape: {weight.shape}"

        pred = torch.log_softmax(pred, dim=2)  # b k c
        weight_sum = (
            calc_weight_sum(weight, self.cfg.loss_weight)
            if weight is not None
            else pred.shape[0]
        )
        loss = self.criterion(pred, target)  # b k c

        if self.model.training:
            for c in range(6):
                loss[..., c] *= self.class_weights[c]

        loss = loss.sum(dim=2)  # b k

        if weight is not None:
            loss *= weight  # b k

        loss = loss.mean(dim=-1)  # b

        if aggregate:
            loss = loss.sum() / weight_sum

        return loss, weight_sum

    @torch.no_grad()
    def pruning_samples(self, batch: dict[str, Tensor], forget_rate: float) -> Tensor:
        if self.teacher_model is None:
            raise ValueError("teacher_model is not set.")

        output = self.teacher_model(batch)
        loss, _ = self._calc_loss(
            output[self.pred_key],
            batch[self.target_key],
            batch[self.weight_key] if self.cfg.distillation.use_loss_weights else None,
            aggregate=False,
        )
        sorted_indices = torch.argsort(loss.data)
        loss_sorted = loss[sorted_indices]

        remain_rate = 1.0 - forget_rate
        num_remain = int(remain_rate * len(loss_sorted))
        indices_to_update = sorted_indices[:num_remain]

        return indices_to_update

    def train_epoch(self, epoch: int):
        self.model.train()

        if self.teacher_model is not None:
            self.teacher_model.eval()

        with tqdm(self.train_loader, unit="step") as pbar:
            for batch in pbar:
                self.optimizer.zero_grad()
                self._move_device(batch)
                with torch.autocast(device_type="cuda", enabled=self.mixed_precision):
                    batch[self.weight_key] **= self.weight_exponent_scheduler.value

                    if self.teacher_model is not None:
                        indices_to_update = self.pruning_samples(
                            batch, self.forget_rate_scheduler.value
                        )
                        self.forget_rate_scheduler.step()

                    output = self.model(batch)
                    loss, weight_sum = self._calc_loss(
                        output[self.pred_key],
                        batch[self.target_key],
                        batch[self.weight_key] if self.cfg.use_loss_weights else None,
                        aggregate=False,
                    )
                    self.weight_exponent_scheduler.step()
                    if self.teacher_model is not None:
                        loss = loss[indices_to_update]
                        weight_sum = (
                            (batch[self.weight_key][indices_to_update].sum().item())
                            if self.cfg.use_loss_weights
                            else len(indices_to_update)
                        )
                    loss = loss.sum() / weight_sum
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
                pbar.set_postfix(
                    {
                        "loss": self._train_loss_meter.mean,
                        "forget_rate": self.forget_rate_scheduler.value,
                        "weight_exponent": self.weight_exponent_scheduler.value,
                    }
                )

    def valid_epoch(self, epoch: int):
        self.model.eval()
        with torch.no_grad():
            with tqdm(self.valid_loader, unit="step") as pbar:
                for batch in pbar:
                    self._move_device(batch)
                    with torch.autocast(device_type="cuda", enabled=True):
                        output = self.model(batch)
                        loss, weight_sum = self._calc_loss(
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
