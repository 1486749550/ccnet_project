import argparse

import torch

from torch.utils.data import DataLoader

from datasets.dataset import BitemporalChangeDataset
from engine.trainer import Trainer
from models.ccnet import CCNet
from utils.logger import setup_logger
from utils.misc import load_yaml
from utils.seed import set_seed


torch.autograd.set_detect_anomaly(True)  #防止梯度消失、爆炸


def build_dataloader(cfg, split: str, is_train: bool) -> DataLoader:
    dataset_cfg = cfg["dataset"]
    root = dataset_cfg["train_root"] if split == "train" else dataset_cfg["val_root"]
    dataset = BitemporalChangeDataset(
        root=root,
        image_size=dataset_cfg["image_size"],
        image_suffix=dataset_cfg["image_suffix"],
        mask_suffix=dataset_cfg["mask_suffix"],
        normalize_mean=dataset_cfg["normalize"]["mean"],
        normalize_std=dataset_cfg["normalize"]["std"],
        is_train=is_train,
    )
    return DataLoader(
        dataset,
        batch_size=cfg["optim"]["batch_size"],
        shuffle=is_train,
        num_workers=cfg["num_workers"],
        pin_memory=True,
        drop_last=is_train,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train CCNet")
    parser.add_argument("--config", type=str, default="configs/ccnet_base.yaml")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    set_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() and cfg["device"] == "cuda" else "cpu")
    logger = setup_logger(cfg["save_dir"])

    train_loader = build_dataloader(cfg, split="train", is_train=True)
    val_loader = build_dataloader(cfg, split="val", is_train=False)

    model = CCNet(
        in_channels=cfg["model"]["in_channels"],
        base_channels=cfg["model"]["base_channels"],
    ).to(device)

    trainer = Trainer(model=model, train_loader=train_loader, val_loader=val_loader, cfg=cfg, logger=logger, device=device)
    trainer.train()


if __name__ == "__main__":
    main()
