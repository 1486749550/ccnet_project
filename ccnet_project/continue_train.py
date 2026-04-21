import argparse

import torch

from engine.trainer import Trainer
from models.ccnet import CCNet
from train import build_dataloader
from utils.checkpoint import load_checkpoint
from utils.logger import setup_logger
from utils.misc import load_yaml
from utils.seed import set_seed


torch.autograd.set_detect_anomaly(True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Continue training CCNet from a checkpoint")
    parser.add_argument("--config", type=str, default="configs/ccnet_base.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    parser.add_argument("--epochs", type=int, default=None, help="Total epochs after resuming")
    parser.add_argument("--batch_size", type=int, default=None, help="Override batch size")
    parser.add_argument("--save_dir", type=str, default=None, help="Override checkpoint output directory")
    parser.add_argument(
        "--weights_only",
        action="store_true",
        help="Load only model weights and reinitialize optimizer/scheduler",
    )
    return parser.parse_args()


def apply_overrides(cfg: dict, args: argparse.Namespace) -> dict:
    if args.lr is not None:
        cfg["optim"]["lr"] = args.lr
    if args.epochs is not None:
        cfg["optim"]["epochs"] = args.epochs
    if args.batch_size is not None:
        cfg["optim"]["batch_size"] = args.batch_size
    if args.save_dir is not None:
        cfg["save_dir"] = args.save_dir
    return cfg


def main() -> None:
    args = parse_args()

    cfg = load_yaml(args.config)
    cfg = apply_overrides(cfg, args)
    set_seed(cfg["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() and cfg["device"] == "cuda" else "cpu")
    logger = setup_logger(cfg["save_dir"])

    checkpoint = load_checkpoint(args.checkpoint, map_location=device)

    train_loader = build_dataloader(cfg, split="train", is_train=True)
    val_loader = build_dataloader(cfg, split="val", is_train=False)

    model = CCNet(
        in_channels=cfg["model"]["in_channels"],
        base_channels=cfg["model"]["base_channels"],
    ).to(device)
    model.load_state_dict(checkpoint["model"])

    start_epoch = 0 if args.weights_only else int(checkpoint.get("epoch", 0))
    best_f1 = -1.0 if args.weights_only else float(checkpoint.get("best_f1", -1.0))
    optimizer_state = None if args.weights_only else checkpoint.get("optimizer")
    scheduler_state = None if args.weights_only else checkpoint.get("scheduler")

    if start_epoch >= cfg["optim"]["epochs"]:
        logger.warning(
            "Checkpoint epoch %d is already >= target epochs %d; no training will run.",
            start_epoch,
            cfg["optim"]["epochs"],
        )

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        cfg=cfg,
        logger=logger,
        device=device,
        start_epoch=start_epoch,
        best_f1=best_f1,
        optimizer_state=optimizer_state,
        scheduler_state=scheduler_state,
        override_lr=args.lr,
    )
    trainer.train()


if __name__ == "__main__":
    main()
