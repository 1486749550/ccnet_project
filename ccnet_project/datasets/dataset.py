from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from datasets.transforms import NormalizeConfig, PairedTransform


class BitemporalChangeDataset(Dataset):
    """
    Bitemporal remote-sensing change detection dataset.

    Returns:
    {
        "img_t1": Tensor,  # [3, H, W]
        "img_t2": Tensor,  # [3, H, W]
        "mask": Tensor,    # [H, W], values 0/1 after transform
        "name": str
    }
    """

    def __init__(
        self,
        root: str,
        image_size: int,
        image_suffix: str | Sequence[str],
        mask_suffix: str | Sequence[str],
        normalize_mean: List[float],
        normalize_std: List[float],
        is_train: bool = True,
    ) -> None:
        self.root = Path(root)
        self.t1_dir, self.t2_dir, self.mask_dir = self._resolve_data_dirs()
        self.image_suffixes = self._normalize_suffixes(image_suffix)
        self.mask_suffixes = self._normalize_suffixes(mask_suffix)
        self.transform = PairedTransform(
            image_size=image_size,
            normalize_cfg=NormalizeConfig(tuple(normalize_mean), tuple(normalize_std)),
            is_train=is_train,
        )
        self.samples = self._scan_samples()

    def _resolve_data_dirs(self) -> tuple[Path, Path, Path]:
        candidates = [
            (self.root / "t1", self.root / "t2", self.root / "mask"),
            (self.root / "A", self.root / "B", self.root / "label"),
        ]
        for t1_dir, t2_dir, mask_dir in candidates:
            if t1_dir.exists() and t2_dir.exists() and mask_dir.exists():
                return t1_dir, t2_dir, mask_dir
        raise AssertionError(
            f"No supported data directory structure found under {self.root}. "
            "Expected t1/t2/mask or A/B/label."
        )

    @staticmethod
    def _normalize_suffixes(suffixes: str | Sequence[str]) -> tuple[str, ...]:
        if isinstance(suffixes, str):
            suffixes = [suffixes]
        normalized = []
        for suffix in suffixes:
            suffix = suffix.strip().lower()
            if not suffix:
                continue
            normalized.append(suffix if suffix.startswith(".") else f".{suffix}")
        assert normalized, "image_suffix/mask_suffix must contain at least one valid suffix"
        return tuple(dict.fromkeys(normalized))

    # def _find_by_stem(self, directory: Path, stem: str, suffixes: tuple[str, ...], label: str) -> Path:
    #     matches = [
    #         path
    #         for path in directory.iterdir()
    #         if path.is_file() and path.stem == stem and path.suffix.lower() in suffixes
    #     ]
    #     assert matches, f"Missing matching {label}: {directory / stem}; supported suffixes: {suffixes}"
    #     assert len(matches) == 1, f"Multiple matching {label} files for stem '{stem}': {matches}"
    #     return matches[0]

    def _find_by_stem(self, directory, stem: str, suffixes: tuple[str, ...], label: str):
        """极速版：O(1) 复杂度，直接拼接路径，拒绝全量遍历"""
        found_paths = []
        
        # 尝试直接拼接后缀名，一击必中
        for suffix in suffixes:
            target_path = directory / f"{stem}{suffix}"
            if target_path.is_file():
                found_paths.append(target_path)
                
        # 保持你原有的安全校验逻辑
        assert found_paths, f"Missing matching {label}: {directory / stem}; supported suffixes: {suffixes}"
        assert len(found_paths) == 1, f"Multiple matching {label} files for stem '{stem}': {found_paths}"
        return found_paths[0]

    def _scan_samples(self) -> List[Dict[str, Path]]:
        assert self.t1_dir.exists(), f"Missing t1 directory: {self.t1_dir}"
        assert self.t2_dir.exists(), f"Missing t2 directory: {self.t2_dir}"
        assert self.mask_dir.exists(), f"Missing mask directory: {self.mask_dir}"

        samples: List[Dict[str, Path]] = []
        t1_paths = [
            path
            for path in self.t1_dir.iterdir()
            if path.is_file() and path.suffix.lower() in self.image_suffixes
        ]
        for t1_path in sorted(t1_paths, key=lambda path: path.name):
            stem = t1_path.stem
            t2_path = self._find_by_stem(self.t2_dir, stem, self.image_suffixes, "img_t2")
            mask_path = self._find_by_stem(self.mask_dir, stem, self.mask_suffixes, "mask")
            samples.append({"name": stem, "img_t1": t1_path, "img_t2": t2_path, "mask": mask_path})

        assert samples, f"No samples found in {self.t1_dir}; supported image suffixes: {self.image_suffixes}"
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def _read_rgb(self, path: Path) -> np.ndarray:
        image = np.array(Image.open(path).convert("RGB"), dtype=np.uint8)
        assert image.ndim == 3 and image.shape[2] == 3, f"Failed to read RGB image: {path}"
        return image

    def _read_mask(self, path: Path) -> np.ndarray:
        mask = np.array(Image.open(path).convert("L"), dtype=np.uint8)
        assert mask.ndim == 2, f"Failed to read mask: {path}"
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
