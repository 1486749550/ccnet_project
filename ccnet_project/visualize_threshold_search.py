import argparse
import csv
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from datasets.dataset import BitemporalChangeDataset
from engine.evaluator import Evaluator
from losses.total_loss import TotalLoss
from models.ccnet import CCNet
from utils.checkpoint import load_checkpoint
from utils.misc import ensure_dir, load_yaml


def build_val_dataloader(cfg: Dict) -> DataLoader:
    """构建验证集 DataLoader。"""
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
    return dataloader


def build_model_and_evaluator(cfg: Dict, checkpoint_path: str, device: torch.device) -> Evaluator:
    """构建模型、加载权重，并返回 Evaluator。"""
    model = CCNet(
        in_channels=cfg["model"]["in_channels"],
        base_channels=cfg["model"]["base_channels"],
    ).to(device)

    checkpoint = load_checkpoint(checkpoint_path, map_location=device)
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
    return evaluator


def save_threshold_metrics_csv(results: Dict[str, Dict[str, float | int]], save_path: str) -> None:
    """将阈值扫描结果保存为 CSV，方便后续分析。"""
    fieldnames = [
        "threshold",
        "precision",
        "recall",
        "f1",
        "accuracy",
        "iou",
        "loss_total",
        "tp",
        "fp",
        "fn",
        "tn",
    ]
    with open(save_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for threshold_key, score_dict in results.items():
            writer.writerow(
                {
                    "threshold": float(threshold_key),
                    "precision": float(score_dict["precision"]),
                    "recall": float(score_dict["recall"]),
                    "f1": float(score_dict["f1"]),
                    "accuracy": float(score_dict["accuracy"]),
                    "iou": float(score_dict["iou"]),
                    "loss_total": float(score_dict["loss_total"]),
                    "tp": int(score_dict["tp"]),
                    "fp": int(score_dict["fp"]),
                    "fn": int(score_dict["fn"]),
                    "tn": int(score_dict["tn"]),
                }
            )


def plot_threshold_curves(results: Dict[str, Dict[str, float | int]], save_path: str) -> str:
    """
    可视化阈值搜索曲线。

    横轴为 threshold，纵轴绘制：
    - F1
    - Precision
    - Recall
    - IoU
    - Accuracy
    """
    thresholds = [float(key) for key in results.keys()]
    precisions = [float(results[key]["precision"]) for key in results.keys()]
    recalls = [float(results[key]["recall"]) for key in results.keys()]
    f1_scores = [float(results[key]["f1"]) for key in results.keys()]
    ious = [float(results[key]["iou"]) for key in results.keys()]
    accuracies = [float(results[key]["accuracy"]) for key in results.keys()]

    best_index = max(range(len(f1_scores)), key=lambda idx: f1_scores[idx])
    best_threshold = thresholds[best_index]
    best_f1 = f1_scores[best_index]

    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, f1_scores, marker="o", linewidth=2.5, label="F1")
    plt.plot(thresholds, precisions, marker="s", linewidth=1.8, label="Precision")
    plt.plot(thresholds, recalls, marker="^", linewidth=1.8, label="Recall")
    plt.plot(thresholds, ious, marker="d", linewidth=1.8, label="IoU")
    plt.plot(thresholds, accuracies, marker="x", linewidth=1.8, label="Accuracy")

    plt.scatter([best_threshold], [best_f1], color="red", s=90, zorder=5)
    plt.annotate(
        f"best threshold={best_threshold:.2f}\nF1={best_f1:.4f}",
        xy=(best_threshold, best_f1),
        xytext=(best_threshold, best_f1 + 0.03),
        arrowprops={"arrowstyle": "->", "lw": 1.2},
        fontsize=10,
    )

    plt.title("Threshold Search on Validation Set")
    plt.xlabel("Threshold")
    plt.ylabel("Score")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    return f"best threshold={best_threshold:.2f}, best f1={best_f1:.6f}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize threshold search for validation F1")
    parser.add_argument("--config", type=str, default="configs/ccnet_base.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=[0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85],
        help="用于阈值搜索的候选列表",
    )
    parser.add_argument("--save_dir", type=str, default="outputs/threshold_search")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    ensure_dir(args.save_dir)
    device = torch.device("cuda" if torch.cuda.is_available() and cfg["device"] == "cuda" else "cpu")

    dataloader = build_val_dataloader(cfg)
    evaluator = build_model_and_evaluator(cfg, args.checkpoint, device)

    thresholds: List[float] = [float(threshold) for threshold in args.thresholds]
    results = evaluator.evaluate_thresholds(dataloader, thresholds)

    save_root = Path(args.save_dir)
    csv_path = save_root / "threshold_metrics.csv"
    fig_path = save_root / "threshold_f1_curve.png"

    save_threshold_metrics_csv(results, str(csv_path))
    summary = plot_threshold_curves(results, str(fig_path))

    print("Threshold search results:")
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

    print(f"\nCSV saved to: {csv_path}")
    print(f"Figure saved to: {fig_path}")
    print(summary)


if __name__ == "__main__":
    main()
