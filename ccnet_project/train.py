import argparse
from pathlib import Path

import torch

from torch.utils.data import DataLoader, Subset

from datasets.dataset import BitemporalChangeDataset
from engine.trainer import Trainer
from models.ccnet import CCNet
from utils.logger import setup_logger
from utils.misc import load_yaml
from utils.seed import set_seed


torch.autograd.set_detect_anomaly(True)  #防止梯度消失、爆炸


def build_dataset(cfg, root: str, is_train: bool) -> BitemporalChangeDataset:
    dataset_cfg = cfg["dataset"]
    return BitemporalChangeDataset(
        root=root,
        image_size=dataset_cfg["image_size"],
        image_suffix=dataset_cfg["image_suffix"],
        mask_suffix=dataset_cfg["mask_suffix"],
        normalize_mean=dataset_cfg["normalize"]["mean"],
        normalize_std=dataset_cfg["normalize"]["std"],
        is_train=is_train,
    )


def build_dataloader(cfg, split: str, is_train: bool) -> DataLoader:
    dataset_cfg = cfg["dataset"]
    root = dataset_cfg["train_root"] if split == "train" else dataset_cfg["val_root"]
    dataset = build_dataset(cfg, root=root, is_train=is_train)
    return DataLoader(
        dataset,
        batch_size=cfg["optim"]["batch_size"],
        shuffle=is_train,
        num_workers=cfg["num_workers"],
        pin_memory=True,
        drop_last=is_train,
    )


def build_train_val_dataloaders(cfg) -> tuple[DataLoader, DataLoader]:
    dataset_cfg = cfg["dataset"]
    train_root = dataset_cfg["train_root"]
    val_root = dataset_cfg.get("val_root")
    use_train_val_split = bool(dataset_cfg.get("use_train_val_split", False))

    if not use_train_val_split and val_root and Path(val_root).exists():
        return (
            build_dataloader(cfg, split="train", is_train=True),
            build_dataloader(cfg, split="val", is_train=False),
        )

    full_train_dataset = build_dataset(cfg, root=train_root, is_train=True)
    full_val_dataset = build_dataset(cfg, root=train_root, is_train=False)
    total_size = len(full_train_dataset)
    assert total_size >= 2, "从训练集划分验证集时，训练样本数至少需要 2 个"

    val_split = float(dataset_cfg.get("val_split", 0.2))
    assert 0.0 < val_split < 1.0, "dataset.val_split 必须在 0 和 1 之间"
    val_size = max(1, min(total_size - 1, round(total_size * val_split)))

    generator = torch.Generator().manual_seed(int(cfg["seed"]))
    indices = torch.randperm(total_size, generator=generator).tolist()
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]

    train_dataset = Subset(full_train_dataset, train_indices)
    val_dataset = Subset(full_val_dataset, val_indices)
    return (
        DataLoader(
            train_dataset,
            batch_size=cfg["optim"]["batch_size"],
            shuffle=True,
            num_workers=cfg["num_workers"],
            pin_memory=True,
            drop_last=True,
        ),
        DataLoader(
            val_dataset,
            batch_size=cfg["optim"]["batch_size"],
            shuffle=False,
            num_workers=cfg["num_workers"],
            pin_memory=True,
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train CCNet")
    parser.add_argument("--config", type=str, default="configs/ccnet_base.yaml")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    set_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() and cfg["device"] == "cuda" else "cpu")
    logger = setup_logger(cfg["save_dir"])

    train_loader, val_loader = build_train_val_dataloaders(cfg)

    model = CCNet(
        in_channels=cfg["model"]["in_channels"],
        base_channels=cfg["model"]["base_channels"],
    ).to(device)

    trainer = Trainer(model=model, train_loader=train_loader, val_loader=val_loader, cfg=cfg, logger=logger, device=device)
    trainer.train()


if __name__ == "__main__":
    main()
