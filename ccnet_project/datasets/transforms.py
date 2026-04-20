import random
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class NormalizeConfig:
    """图像归一化参数。"""

    mean: Tuple[float, float, float]
    std: Tuple[float, float, float]


class PairedTransform:
    """
    双时相同步增强。

    所有几何操作都必须对 img_t1、img_t2、mask 严格同步，
    否则会破坏变化检测任务最关键的像素级对齐关系。
    """

    def __init__(self, image_size: int, normalize_cfg: NormalizeConfig, is_train: bool = True) -> None:
        self.image_size = image_size
        self.normalize_cfg = normalize_cfg
        self.is_train = is_train
        self.mean = torch.tensor(normalize_cfg.mean, dtype=torch.float32).view(3, 1, 1)
        self.std = torch.tensor(normalize_cfg.std, dtype=torch.float32).view(3, 1, 1)

    def _to_tensor(self, image: np.ndarray) -> torch.Tensor:
        """将 HWC 图像转为 CHW tensor。"""
        assert image.ndim == 3, f"图像必须是 HWC 格式，当前为: {image.shape}"
        tensor = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
        assert tensor.ndim == 3 and tensor.shape[0] == 3, f"图像 tensor 格式错误，当前为: {tensor.shape}"
        return tensor

    def _mask_to_tensor(self, mask: np.ndarray) -> torch.Tensor:
        """将标签图转为 [H, W]，取值为 0/1。"""
        assert mask.ndim == 2, f"标签必须是二维，当前为: {mask.shape}"
        tensor = torch.from_numpy((mask > 127).astype(np.int64))
        assert tensor.ndim == 2, f"mask tensor 维度错误，当前为: {tensor.shape}"
        return tensor

    def _resize_img(self, image: torch.Tensor) -> torch.Tensor:
        image = F.interpolate(image.unsqueeze(0), size=(self.image_size, self.image_size), mode="bilinear", align_corners=False).squeeze(0)
        assert image.shape[-2:] == (self.image_size, self.image_size), "图像 resize 后尺寸错误"
        return image

    def _resize_mask(self, mask: torch.Tensor) -> torch.Tensor:
        resized = F.interpolate(mask.unsqueeze(0).unsqueeze(0).float(), size=(self.image_size, self.image_size), mode="nearest").squeeze(0).squeeze(0).long()
        assert resized.shape == (self.image_size, self.image_size), "mask resize 后尺寸错误"
        return resized

    def _random_flip(self, img_t1: torch.Tensor, img_t2: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """随机水平/垂直翻转。"""
        if random.random() < 0.5:
            img_t1 = torch.flip(img_t1, dims=[2])
            img_t2 = torch.flip(img_t2, dims=[2])
            mask = torch.flip(mask, dims=[1])
        if random.random() < 0.5:
            img_t1 = torch.flip(img_t1, dims=[1])
            img_t2 = torch.flip(img_t2, dims=[1])
            mask = torch.flip(mask, dims=[0])
        return img_t1, img_t2, mask

    def _random_rotate90(self, img_t1: torch.Tensor, img_t2: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """随机 90 度旋转。"""
        k = random.randint(0, 3)
        img_t1 = torch.rot90(img_t1, k, dims=[1, 2])
        img_t2 = torch.rot90(img_t2, k, dims=[1, 2])
        mask = torch.rot90(mask, k, dims=[0, 1])
        return img_t1, img_t2, mask

    def _random_shift(self, img_t1: torch.Tensor, img_t2: torch.Tensor, mask: torch.Tensor, max_shift: int = 8) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        随机平移。

        这里采用等价几何增强方案：对三者做同样的像素平移，并在边界处零填充。
        """
        shift_y = random.randint(-max_shift, max_shift)
        shift_x = random.randint(-max_shift, max_shift)
        img_t1 = torch.roll(img_t1, shifts=(shift_y, shift_x), dims=(1, 2))
        img_t2 = torch.roll(img_t2, shifts=(shift_y, shift_x), dims=(1, 2))
        mask = torch.roll(mask, shifts=(shift_y, shift_x), dims=(0, 1))
        return img_t1, img_t2, mask

    def _normalize(self, image: torch.Tensor) -> torch.Tensor:
        return (image - self.mean) / self.std

    def __call__(self, sample: Dict[str, np.ndarray]) -> Dict[str, torch.Tensor]:
        img_t1 = self._resize_img(self._to_tensor(sample["img_t1"]))
        img_t2 = self._resize_img(self._to_tensor(sample["img_t2"]))
        mask = self._resize_mask(self._mask_to_tensor(sample["mask"]))

        if self.is_train:
            img_t1, img_t2, mask = self._random_flip(img_t1, img_t2, mask)
            img_t1, img_t2, mask = self._random_rotate90(img_t1, img_t2, mask)
            img_t1, img_t2, mask = self._random_shift(img_t1, img_t2, mask)

        img_t1 = self._normalize(img_t1)
        img_t2 = self._normalize(img_t2)

        output = {"img_t1": img_t1, "img_t2": img_t2, "mask": mask}
        assert output["img_t1"].shape == output["img_t2"].shape, "双时相图像尺寸必须一致"
        assert output["mask"].ndim == 2, "mask 必须保持 [H, W]"
        return output
