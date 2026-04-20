import argparse

import torch
from torch.utils.data import DataLoader

from datasets.dataset import BitemporalChangeDataset
from engine.inferencer import Inferencer
from models.ccnet import CCNet
from utils.checkpoint import load_checkpoint
from utils.misc import ensure_dir, load_yaml


def main() -> None:
    parser = argparse.ArgumentParser(description="Infer with CCNet")
    parser.add_argument("--config", type=str, default="configs/ccnet_base.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--save_dir", type=str, default="outputs/infer")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    ensure_dir(args.save_dir)
    device = torch.device("cuda" if torch.cuda.is_available() and cfg["device"] == "cuda" else "cpu")

    dataset_cfg = cfg["dataset"]
    dataset = BitemporalChangeDataset(
        root=dataset_cfg["test_root"],
        image_size=dataset_cfg["image_size"],
        image_suffix=dataset_cfg["image_suffix"],
        mask_suffix=dataset_cfg["mask_suffix"],
        normalize_mean=dataset_cfg["normalize"]["mean"],
        normalize_std=dataset_cfg["normalize"]["std"],
        is_train=False,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=cfg["optim"]["batch_size"],
        shuffle=False,
        num_workers=cfg["num_workers"],
        pin_memory=True,
    )

    model = CCNet(
        in_channels=cfg["model"]["in_channels"],
        base_channels=cfg["model"]["base_channels"],
    ).to(device)
    checkpoint = load_checkpoint(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model"])

    inferencer = Inferencer(model=model, device=device)
    inferencer.run(dataloader, args.save_dir)


if __name__ == "__main__":
    main()
