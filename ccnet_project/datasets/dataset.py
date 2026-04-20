from pathlib import Path
from typing import Dict, List

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from datasets.transforms import NormalizeConfig, PairedTransform


class BitemporalChangeDataset(Dataset):
    """
    双时相遥感变化检测数据集。

    返回格式严格为：
    {
        "img_t1": Tensor,  # [3, H, W]
        "img_t2": Tensor,  # [3, H, W]
        "mask": Tensor,    # [H, W]，取值 0/1
        "name": str
    }
    """

    def __init__(
        self,
        root: str,
        image_size: int,
        image_suffix: str,
        mask_suffix: str,
        normalize_mean: List[float],
        normalize_std: List[float],
        is_train: bool = True,
    ) -> None:
        self.root = Path(root)
        self.t1_dir, self.t2_dir, self.mask_dir = self._resolve_data_dirs()
        self.image_suffix = image_suffix
        self.mask_suffix = mask_suffix
        self.transform = PairedTransform(
            image_size=image_size,
            normalize_cfg=NormalizeConfig(tuple(normalize_mean), tuple(normalize_std)),
            is_train=is_train,
        )
        self.samples = self._scan_samples()

    def _resolve_data_dirs(self) -> tuple[Path, Path, Path]:
        """
        自动解析双时相数据目录。

        优先支持两种常见组织方式：
        1. t1 / t2 / mask
        2. A / B / label
        """
        candidates = [
            (self.root / "t1", self.root / "t2", self.root / "mask"),
            (self.root / "A", self.root / "B", self.root / "label"),
        ]
        for t1_dir, t2_dir, mask_dir in candidates:
            if t1_dir.exists() and t2_dir.exists() and mask_dir.exists():
                return t1_dir, t2_dir, mask_dir
        raise AssertionError(
            f"在 {self.root} 下没有找到支持的数据目录结构。"
            f" 需要存在 t1/t2/mask 或 A/B/label，当前目录为: {self.root}"
        )

    def _scan_samples(self) -> List[Dict[str, Path]]:
        assert self.t1_dir.exists(), f"缺少 t1 目录: {self.t1_dir}"
        assert self.t2_dir.exists(), f"缺少 t2 目录: {self.t2_dir}"
        assert self.mask_dir.exists(), f"缺少 mask 目录: {self.mask_dir}"

        samples: List[Dict[str, Path]] = []
        for t1_path in sorted(self.t1_dir.glob(f"*{self.image_suffix}")):
            stem = t1_path.stem
            t2_path = self.t2_dir / f"{stem}{self.image_suffix}"
            mask_path = self.mask_dir / f"{stem}{self.mask_suffix}"
            assert t2_path.exists(), f"缺少对应 img_t2: {t2_path}"
            assert mask_path.exists(), f"缺少对应 mask: {mask_path}"
            samples.append({"name": stem, "img_t1": t1_path, "img_t2": t2_path, "mask": mask_path})
        assert len(samples) > 0, f"在 {self.t1_dir} 中没有发现任何样本"
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def _read_rgb(self, path: Path) -> np.ndarray:
        image = np.array(Image.open(path).convert("RGB"), dtype=np.uint8)
        assert image.ndim == 3 and image.shape[2] == 3, f"读取 RGB 图像失败: {path}"
        return image

    def _read_mask(self, path: Path) -> np.ndarray:
        mask = np.array(Image.open(path).convert("L"), dtype=np.uint8)
        assert mask.ndim == 2, f"读取 mask 失败: {path}"
        return mask

    def __getitem__(self, index: int) -> Dict[str, object]:
        sample_info = self.samples[index]
        raw = {
            "img_t1": self._read_rgb(sample_info["img_t1"]),
            "img_t2": self._read_rgb(sample_info["img_t2"]),
            "mask": self._read_mask(sample_info["mask"]),
        }
        output = self.transform(raw)
        output["name"] = sample_info["name"]
        return output


ChangeDetectionDataset = BitemporalChangeDataset
