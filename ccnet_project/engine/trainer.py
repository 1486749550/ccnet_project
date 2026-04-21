from pathlib import Path
from typing import Any, Dict

import torch
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from engine.evaluator import Evaluator
from losses.total_loss import TotalLoss
from utils.checkpoint import save_checkpoint
from utils.misc import AverageMeter


class Trainer:
    """标准 epoch 训练器。"""

    def __init__(
        self,
        model: torch.nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        cfg: Dict,
        logger,
        device: torch.device,
        start_epoch: int = 0,
        best_f1: float = -1.0,
        optimizer_state: Dict[str, Any] | None = None,
        scheduler_state: Dict[str, Any] | None = None,
        override_lr: float | None = None,
    ) -> None:
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.cfg = cfg
        self.logger = logger
        self.device = device

        self.criterion = TotalLoss(
            lambda_sep=cfg["loss"]["lambda_sep"],
            lambda_sim=cfg["loss"]["lambda_sim"],
            lambda_rst=cfg["loss"]["lambda_rst"],
            tversky_alpha=cfg["loss"].get("tversky_alpha", 0.8),
            tversky_beta=cfg["loss"].get("tversky_beta", 0.2),
            tversky_weight=cfg["loss"].get("tversky_weight", 1.0),
        ).to(device)
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=cfg["optim"]["lr"],
            weight_decay=cfg["optim"]["weight_decay"],
            betas=tuple(cfg["optim"]["betas"]),
        )
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=cfg["optim"]["epochs"],
            eta_min=cfg["scheduler"]["min_lr"],
        )
        if optimizer_state is not None:
            self.optimizer.load_state_dict(optimizer_state)
        if scheduler_state is not None:
            self.scheduler.load_state_dict(scheduler_state)
            self.scheduler.T_max = cfg["optim"]["epochs"]
        if override_lr is not None:
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = override_lr
                param_group["initial_lr"] = override_lr
            self.scheduler.base_lrs = [override_lr for _ in self.scheduler.base_lrs]
            self.scheduler._last_lr = [override_lr for _ in self.scheduler._last_lr]
        self.scaler = GradScaler(enabled=cfg["optim"]["amp"] and device.type == "cuda")
        self.evaluator = Evaluator(
            model=self.model,
            criterion=self.criterion,
            device=device,
            threshold=cfg["inference"].get("threshold", 0.75),
        )
        self.start_epoch = int(start_epoch)
        self.best_f1 = float(best_f1)

    def train(self) -> None:
        save_dir = Path(self.cfg["save_dir"])
        save_dir.mkdir(parents=True, exist_ok=True)

        for epoch in range(self.start_epoch + 1, self.cfg["optim"]["epochs"] + 1):
            train_stats = self.train_one_epoch(epoch)
            val_stats = self.evaluator.evaluate(self.val_loader)
            self.scheduler.step()

            self.logger.info(
                "Epoch [%d/%d] total=%.6f cd=%.6f sep=%.6f sim=%.6f rst=%.6f | val_f1=%.4f val_iou=%.4f val_acc=%.4f threshold=%.2f",
                epoch,
                self.cfg["optim"]["epochs"],
                train_stats["loss_total"],
                train_stats["loss_cd"],
                train_stats["loss_sep"],
                train_stats["loss_sim"],
                train_stats["loss_rst"],
                val_stats["f1"],
                val_stats["iou"],
                val_stats["accuracy"],
                val_stats["threshold"],
            )

            state = {
                "epoch": epoch,
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(),
                "best_f1": self.best_f1,
                "config": self.cfg,
            }
            save_checkpoint(state, str(save_dir / "latest.pth"))

            if float(val_stats["f1"]) > self.best_f1:
                self.best_f1 = float(val_stats["f1"])
                state["best_f1"] = self.best_f1
                save_checkpoint(state, str(save_dir / "best_f1.pth"))

    def train_one_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.train()
        meters = {name: AverageMeter() for name in ["loss_total", "loss_cd", "loss_sep", "loss_sim", "loss_rst"]}

        for step, batch in enumerate(self.train_loader, start=1):
            batch = self._move_to_device(batch)
            self.optimizer.zero_grad(set_to_none=True)

            with autocast(device_type=self.device.type, enabled=self.scaler.is_enabled()):
                outputs = self.model(batch["img_t1"], batch["img_t2"])
                losses = self.criterion(outputs, batch["img_t1"], batch["img_t2"], batch["mask"])
                total_loss = losses["loss_total"]

            self.scaler.scale(total_loss).backward()
            if self.cfg["optim"]["grad_clip"] > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg["optim"]["grad_clip"])
            self.scaler.step(self.optimizer)
            self.scaler.update()


            batch_size = batch["img_t1"].shape[0]
            for key in meters:
                meters[key].update(float(losses[key].item()), batch_size)

            if step % self.cfg["optim"]["log_interval"] == 0:
                self.logger.info(
                    "Epoch %d Step %d/%d total=%.6f cd=%.6f sep=%.6f sim=%.6f rst=%.6f",
                    epoch,
                    step,
                    len(self.train_loader),
                    losses["loss_total"].item(),
                    losses["loss_cd"].item(),
                    losses["loss_sep"].item(),
                    losses["loss_sim"].item(),
                    losses["loss_rst"].item(),
                )

        return {key: meter.avg for key, meter in meters.items()}

    def _move_to_device(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        batch["img_t1"] = batch["img_t1"].to(self.device, non_blocking=True)
        batch["img_t2"] = batch["img_t2"].to(self.device, non_blocking=True)
        batch["mask"] = batch["mask"].to(self.device, non_blocking=True)
        return batch
