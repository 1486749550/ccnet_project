from pathlib import Path
from typing import Dict

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader


class Inferencer:
    """推理器，保存概率图与最终二值变化图。"""

    def __init__(self, model: torch.nn.Module, device: torch.device) -> None:
        self.model = model
        self.device = device

    @torch.no_grad()
    def run(self, dataloader: DataLoader, save_dir: str) -> None:
        self.model.eval()
        save_root = Path(save_dir)
        prob_dir = save_root / "prob"
        pred_dir = save_root / "pred"
        prob_dir.mkdir(parents=True, exist_ok=True)
        pred_dir.mkdir(parents=True, exist_ok=True)

        for batch in dataloader:
            batch = self._move_to_device(batch)
            outputs = self.model(batch["img_t1"], batch["img_t2"])
            probs = torch.softmax(outputs["change_logits"], dim=1)[:, 1].cpu().numpy()
            preds = outputs["pred_mask"].cpu().numpy()
            for i, name in enumerate(batch["name"]):
                prob_img = (probs[i] * 255.0).clip(0, 255).astype(np.uint8)
                pred_img = (preds[i] * 255).astype(np.uint8)
                Image.fromarray(prob_img).save(prob_dir / f"{name}.png")
                Image.fromarray(pred_img).save(pred_dir / f"{name}.png")

    def _move_to_device(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        batch["img_t1"] = batch["img_t1"].to(self.device, non_blocking=True)
        batch["img_t2"] = batch["img_t2"].to(self.device, non_blocking=True)
        return batch
