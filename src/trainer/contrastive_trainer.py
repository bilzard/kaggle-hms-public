from typing import cast

import numpy as np
import torch
import torch.nn as nn
from hydra.utils import instantiate
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import get_cosine_schedule_with_warmup

from src.callback.base import Callback
from src.config import ContrastiveConfig, TrainerConfig
from src.scheduler import GaussianRampUpScheduler, LinearScheduler
from src.train_util import AverageMeter, get_lr_params
from src.trainer.base import BaseTrainer
from src.trainer.util import calc_weight_sum


class ContrastiveTrainer(BaseTrainer):
    """
    以下の3つをlossとして加算する
    1. 1d model の予測値
    2. 2d model の予測値
    3. 1d/2dの予測値のsymmetric kl divergence

    さらに、上記のlossを使ってbatch内のサンプルを間引きする。
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
        self._train_loss_meter_eeg = AverageMeter()
        self._train_loss_meter_spec = AverageMeter()
        self._train_loss_meter_contrastive = AverageMeter()

        self._valid_loss_meter = AverageMeter()
        assert len(cfg.class_weights) == 6
        self.class_weights = np.array(cfg.class_weights) ** cfg.class_weight_exponent
        self.class_weights /= self.class_weights.sum()
        self.class_weights *= 6

        print(f"** class_weights: {self.class_weights}")

        self.configure_optimizers()
        self.configure_schedulers()
        self.clear_log()
        self.log_architecture()

    def log_architecture(self):
        self.write_log("Optimizer", str(self.optimizer))
        self.write_log("Train dataset", str(self.train_loader.dataset))
        self.write_log("Valid dataset", str(self.valid_loader.dataset))
        self.write_log("Model:", str(self.model))

    def configure_optimizers(self):
        cfg = self.cfg
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

    def configure_schedulers(self):
        cfg = self.cfg
        max_steps = len(self.train_loader) * self.epochs
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_training_steps=max_steps,
            num_warmup_steps=int(max_steps * cfg.scheduler.warmup_ratio),
            num_cycles=0.5,
        )
        self.min_weight_scheduler = LinearScheduler(
            initial_value=cfg.label.schedule.min_weight.initial_value,
            target_step=len(self.train_loader)
            * cfg.label.schedule.min_weight.target_epoch,
            schedule_start_step=len(self.train_loader)
            * cfg.label.schedule.min_weight.schedule_start_epoch,
            target_value=cfg.label.schedule.min_weight.target_value,
        )
        ssl_config = cast(ContrastiveConfig, cfg.ssl)
        self.contrastive_weight_scheduler = GaussianRampUpScheduler(
            target_step=len(self.train_loader)
            * ssl_config.weight_schedule.target_epoch,
            target_value=ssl_config.weight_schedule.target_value,
            schedule_start_step=len(self.train_loader)
            * ssl_config.weight_schedule.schedule_start_epoch,
            initial_value=ssl_config.weight_schedule.initial_value,
        )
        print(
            "* min_weight: {} -> {} (step: {} -> {})".format(
                self.min_weight_scheduler.initial_value,
                self.min_weight_scheduler.target_value,
                self.min_weight_scheduler.schedule_start_step,
                self.min_weight_scheduler.target_step,
            )
        )
        print(
            "* contrastive_weight: 0 -> {} (step: {} -> {})".format(
                self.contrastive_weight_scheduler.target_value,
                self.contrastive_weight_scheduler.schedule_start_step,
                self.contrastive_weight_scheduler.target_step,
            )
        )

    def update_scheduler(self):
        self.scheduler.step()
        self.min_weight_scheduler.step()
        self.contrastive_weight_scheduler.step()

    def fit(self):
        for epoch in range(self.epochs):
            self._train_loss_meter.reset()
            self._train_loss_meter_eeg.reset()
            self._train_loss_meter_spec.reset()
            self._train_loss_meter_contrastive.reset()

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
        softmax_target: bool = False,
    ) -> tuple[torch.Tensor, float]:
        """
        pred: b k c
        target: b k c
        weight: b k
        """
        pred = torch.log_softmax(pred, dim=2)
        if softmax_target:
            target = torch.softmax(target, dim=2)

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
            loss *= weight

        loss = loss.mean(dim=1)  # b

        if aggregate:
            loss = loss.sum() / weight_sum

        return loss, weight_sum

    def train_epoch(self, epoch: int):
        self.model.train()

        if self.teacher_model is not None:
            self.teacher_model.eval()

        with tqdm(self.train_loader, unit="step") as pbar:
            for batch in pbar:
                self.optimizer.zero_grad()
                self._move_device(batch)
                with torch.autocast(device_type="cuda", enabled=self.mixed_precision):
                    output = self.model(batch)

                    weight = batch[self.weight_key]
                    target = batch[self.target_key]
                    logit_eeg = output["logit_eeg"]
                    logit_spec = output["logit_spec"]

                    # min_weight でフィルタリング
                    valid_indices = torch.where(
                        (weight.mean(dim=1) > self.min_weight_scheduler.value)
                    )[0]
                    if len(valid_indices) == 0:
                        continue

                    target = target[valid_indices]
                    logit_eeg = logit_eeg[valid_indices]
                    logit_spec = logit_spec[valid_indices]
                    weight = weight[valid_indices]

                    loss_eeg, _ = self._calc_loss(
                        logit_eeg,
                        target,
                        weight if self.cfg.use_loss_weights else None,
                        aggregate=False,
                    )
                    loss_spec, _ = self._calc_loss(
                        logit_spec,
                        target,
                        weight if self.cfg.use_loss_weights else None,
                        aggregate=False,
                    )
                    loss_supervised = (loss_eeg + loss_spec) / 2.0
                    weight_sum = calc_weight_sum(weight, self.cfg.loss_weight)
                    loss_supervised = loss_supervised.sum() / weight_sum
                    loss = loss_supervised

                    if self.contrastive_weight_scheduler.value > 0:
                        loss_contrastive_1, _ = self._calc_loss(
                            logit_eeg, logit_spec, softmax_target=True, aggregate=True
                        )
                        loss_contrastive_2, _ = self._calc_loss(
                            logit_spec, logit_eeg, softmax_target=True, aggregate=True
                        )
                        loss_contrastive = (
                            loss_contrastive_1 + loss_contrastive_2
                        ) / 2.0
                        weight_sum_contrastive = weight.shape[0]
                        loss += (
                            self.contrastive_weight_scheduler.value * loss_contrastive
                        )

                    with torch.no_grad():
                        self._train_loss_meter.update(
                            loss_supervised.item(), weight_sum
                        )
                        self._train_loss_meter_eeg.update(
                            loss_eeg.sum().item() / weight_sum,
                            weight_sum,
                        )
                        self._train_loss_meter_spec.update(
                            loss_spec.sum().item() / weight_sum,
                            weight_sum,
                        )
                        if self.contrastive_weight_scheduler.value > 0:
                            self._train_loss_meter_contrastive.update(
                                loss_contrastive.item(), weight_sum_contrastive
                            )

                if self.scaler is not None:
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    self.optimizer.step()
                self.update_scheduler()

                for callback in self.callbacks:
                    callback.on_train_step_end(
                        self, batch, output, self._train_loss_meter.mean
                    )
                pbar.set_postfix(
                    {
                        "loss": self._train_loss_meter.mean,
                        "loss_eeg": self._train_loss_meter_eeg.mean,
                        "loss_spec": self._train_loss_meter_spec.mean,
                        "loss_contrastive": self._train_loss_meter_contrastive.mean,
                        "min_weight": self.min_weight_scheduler.value,
                        "contrastive_weight": self.contrastive_weight_scheduler.value,
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
