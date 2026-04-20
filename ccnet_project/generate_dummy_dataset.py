import argparse
import math
from pathlib import Path
from typing import Tuple

import numpy as np
from PIL import Image


class DummyChangeDatasetGenerator:
    """
    伪双时相变化检测数据集生成器。

    设计目标：
    1. 在空目录下快速生成可训练、可验证、可推理的最小数据集；
    2. 同时构造“真实变化”和“风格扰动导致的伪变化风险”，便于检验 CCNet 的训练流程；
    3. 输出目录严格匹配当前工程默认约定：
       data/
         train/{t1,t2,mask}
         val/{t1,t2,mask}
         test/{t1,t2,mask}
    """

    def __init__(self, root: str, image_size: int, seed: int = 42) -> None:
        self.root = Path(root)
        self.image_size = image_size
        self.rng = np.random.default_rng(seed)

    def _ensure_split_dirs(self, split: str) -> Tuple[Path, Path, Path]:
        """为指定数据划分创建 t1/t2/mask 三个目录。"""
        split_root = self.root / split
        t1_dir = split_root / "t1"
        t2_dir = split_root / "t2"
        mask_dir = split_root / "mask"
        t1_dir.mkdir(parents=True, exist_ok=True)
        t2_dir.mkdir(parents=True, exist_ok=True)
        mask_dir.mkdir(parents=True, exist_ok=True)
        return t1_dir, t2_dir, mask_dir

    def _make_background(self) -> np.ndarray:
        """
        生成基础背景纹理。

        这里采用工程化合成方案：
        - 用低频梯度模拟大范围地表反射变化
        - 用噪声和条带模拟遥感影像中的纹理起伏
        """
        h = w = self.image_size
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        xx = xx / max(w - 1, 1)
        yy = yy / max(h - 1, 1)

        base_r = 70 + 90 * xx + 20 * np.sin(2 * math.pi * yy)
        base_g = 80 + 70 * yy + 15 * np.cos(2 * math.pi * xx)
        base_b = 60 + 60 * (1 - xx) + 20 * np.sin(2 * math.pi * (xx + yy))
        image = np.stack([base_r, base_g, base_b], axis=2)

        texture = self.rng.normal(loc=0.0, scale=10.0, size=(h, w, 3))
        stripe = 8.0 * np.sin(2 * math.pi * xx[..., None] * self.rng.integers(2, 6))
        image = image + texture + stripe
        return np.clip(image, 0, 255).astype(np.uint8)

    def _draw_rectangle(self, image: np.ndarray, mask: np.ndarray) -> None:
        """绘制矩形变化区域。"""
        h, w, _ = image.shape
        rect_w = int(self.rng.integers(w // 10, w // 4))
        rect_h = int(self.rng.integers(h // 10, h // 4))
        x1 = int(self.rng.integers(0, max(w - rect_w, 1)))
        y1 = int(self.rng.integers(0, max(h - rect_h, 1)))
        x2 = min(x1 + rect_w, w)
        y2 = min(y1 + rect_h, h)
        color = self.rng.integers(20, 235, size=(3,), dtype=np.uint8)
        image[y1:y2, x1:x2] = color
        mask[y1:y2, x1:x2] = 255

    def _draw_circle(self, image: np.ndarray, mask: np.ndarray) -> None:
        """绘制圆形变化区域。"""
        h, w, _ = image.shape
        radius = int(self.rng.integers(min(h, w) // 12, min(h, w) // 6))
        cx = int(self.rng.integers(radius, max(w - radius, radius + 1)))
        cy = int(self.rng.integers(radius, max(h - radius, radius + 1)))
        yy, xx = np.mgrid[0:h, 0:w]
        region = (xx - cx) ** 2 + (yy - cy) ** 2 <= radius ** 2
        color = self.rng.integers(20, 235, size=(3,), dtype=np.uint8)
        image[region] = color
        mask[region] = 255

    def _apply_real_changes(self, image_t2: np.ndarray) -> np.ndarray:
        """
        在 t2 上叠加真实变化区域，并返回二值 mask。

        变化区域会被明显重绘，这样训练时可以学到真实的地物变化。
        """
        mask = np.zeros((self.image_size, self.image_size), dtype=np.uint8)
        num_shapes = int(self.rng.integers(1, 4))
        for _ in range(num_shapes):
            if self.rng.random() < 0.5:
                self._draw_rectangle(image_t2, mask)
            else:
                self._draw_circle(image_t2, mask)
        return mask

    def _apply_style_shift(self, image: np.ndarray) -> np.ndarray:
        """
        施加整图风格扰动。

        这部分不修改 mask，用来模拟光照、色偏、薄云或成像条件变化，
        也就是 CCNet 希望抑制的“伪变化”来源。
        """
        image_f = image.astype(np.float32)

        brightness = float(self.rng.uniform(0.85, 1.15))
        contrast = float(self.rng.uniform(0.85, 1.15))
        color_bias = self.rng.uniform(-15.0, 15.0, size=(1, 1, 3)).astype(np.float32)

        image_mean = image_f.mean(axis=(0, 1), keepdims=True)
        image_f = (image_f - image_mean) * contrast + image_mean
        image_f = image_f * brightness + color_bias

        haze_strength = float(self.rng.uniform(0.0, 0.12))
        if haze_strength > 0:
            haze = np.full_like(image_f, 255.0)
            image_f = image_f * (1.0 - haze_strength) + haze * haze_strength

        return np.clip(image_f, 0, 255).astype(np.uint8)

    def _make_pair(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """生成一对双时相图像及其变化标签。"""
        img_t1 = self._make_background()
        img_t2 = img_t1.copy()
        mask = self._apply_real_changes(img_t2)
        img_t2 = self._apply_style_shift(img_t2)
        return img_t1, img_t2, mask

    def _save_sample(self, split: str, index: int) -> None:
        """保存单个样本。"""
        t1_dir, t2_dir, mask_dir = self._ensure_split_dirs(split)
        img_t1, img_t2, mask = self._make_pair()
        name = f"{index:04d}.png"
        Image.fromarray(img_t1).save(t1_dir / name)
        Image.fromarray(img_t2).save(t2_dir / name)
        Image.fromarray(mask).save(mask_dir / name)

    def generate_split(self, split: str, num_samples: int) -> None:
        """生成一个数据划分。"""
        assert num_samples > 0, f"{split} 样本数必须大于 0，当前为: {num_samples}"
        for index in range(num_samples):
            self._save_sample(split, index)

    def generate_all(self, train_samples: int, val_samples: int, test_samples: int) -> None:
        """一次性生成 train/val/test 全部数据。"""
        self.generate_split("train", train_samples)
        self.generate_split("val", val_samples)
        self.generate_split("test", test_samples)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate dummy dataset for CCNet")
    parser.add_argument("--root", type=str, default="data", help="输出数据根目录")
    parser.add_argument("--image_size", type=int, default=256, help="图像尺寸，必须能被 16 整除")
    parser.add_argument("--train_samples", type=int, default=16, help="训练样本数")
    parser.add_argument("--val_samples", type=int, default=4, help="验证样本数")
    parser.add_argument("--test_samples", type=int, default=4, help="测试样本数")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    args = parser.parse_args()

    assert args.image_size % 16 == 0, f"image_size 必须能被 16 整除，当前为: {args.image_size}"
    generator = DummyChangeDatasetGenerator(root=args.root, image_size=args.image_size, seed=args.seed)
    generator.generate_all(
        train_samples=args.train_samples,
        val_samples=args.val_samples,
        test_samples=args.test_samples,
    )

    print(f"dummy dataset generated at: {Path(args.root).resolve()}")
    print(f"train samples: {args.train_samples}")
    print(f"val samples: {args.val_samples}")
    print(f"test samples: {args.test_samples}")


if __name__ == "__main__":
    main()
