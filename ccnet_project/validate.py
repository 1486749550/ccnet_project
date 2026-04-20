import argparse
from typing import List

import torch
from torch.utils.data import DataLoader

from datasets.dataset import BitemporalChangeDataset
from engine.evaluator import Evaluator
from losses.total_loss import TotalLoss
from models.ccnet import CCNet
from utils.checkpoint import load_checkpoint
from utils.misc import load_yaml


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate CCNet")
    parser.add_argument("--config", type=str, default="configs/ccnet_base.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--threshold", type=float, default=None, help="验证时使用的二值化阈值，默认读取配置或使用 0.75")
    parser.add_argument(
        "--auto_threshold",
        action="store_true",
        help="是否自动扫描多个阈值并选择 F1-Score 最高的结果",
    )
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=[0.5, 0.6, 0.7, 0.8],
        help="自动寻优时要扫描的阈值列表",
    )
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() and cfg["device"] == "cuda" else "cpu")

    dataset_cfg = cfg["dataset"]
    dataset = BitemporalChangeDataset(
        root=dataset_cfg["val_root"],
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

    criterion = TotalLoss(
        lambda_sep=cfg["loss"]["lambda_sep"],
        lambda_sim=cfg["loss"]["lambda_sim"],
        lambda_rst=cfg["loss"]["lambda_rst"],
        tversky_alpha=cfg["loss"].get("tversky_alpha", 0.8),
        tversky_beta=cfg["loss"].get("tversky_beta", 0.2),
        tversky_weight=cfg["loss"].get("tversky_weight", 1.0),
    ).to(device)
    evaluator = Evaluator(
        model=model,
        criterion=criterion,
        device=device,
        threshold=cfg["inference"].get("threshold", 0.75),
    )

    if args.auto_threshold:
        candidate_thresholds: List[float] = [float(threshold) for threshold in args.thresholds]
        results = evaluator.evaluate_thresholds(dataloader, candidate_thresholds)
        best_key = max(results.keys(), key=lambda key: float(results[key]["f1"]))

        print("Auto threshold search results:")
        for threshold_key, score_dict in results.items():
            print(
                f"threshold={threshold_key} "
                f"precision={score_dict['precision']:.6f} "
                f"recall={score_dict['recall']:.6f} "
                f"f1={score_dict['f1']:.6f} "
                f"accuracy={score_dict['accuracy']:.6f} "
                f"iou={score_dict['iou']:.6f} "
                f"tp={score_dict['tp']} fp={score_dict['fp']} fn={score_dict['fn']} tn={score_dict['tn']}"
            )

        print("\nBest threshold result:")
        for key, value in results[best_key].items():
            print(f"{key}: {value}")
    else:
        active_threshold = args.threshold if args.threshold is not None else cfg["inference"].get("threshold", 0.75)
        scores = evaluator.evaluate(dataloader, threshold=active_threshold)
        for key, value in scores.items():
            print(f"{key}: {value}")


if __name__ == "__main__":
    main()
