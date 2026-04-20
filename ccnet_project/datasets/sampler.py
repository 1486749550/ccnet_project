from typing import Iterator, List

import torch
from torch.utils.data import Sampler


class AlternatingSubsetSampler(Sampler[int]):
    """
    一个轻量级采样器示例。

    当数据集存在类别不均衡时，可以通过传入包含“变化样本”索引的列表，
    让采样过程在变化样本与全体样本之间交替，提高 batch 内正样本出现概率。
    如果没有提供 change_indices，则退化为普通随机采样。
    """

    def __init__(self, dataset_size: int, change_indices: List[int] | None = None) -> None:
        self.dataset_size = dataset_size
        self.change_indices = change_indices or []

    def __iter__(self) -> Iterator[int]:
        all_indices = torch.randperm(self.dataset_size).tolist()
        if not self.change_indices:
            return iter(all_indices)

        change_indices = self.change_indices.copy()
        perm = torch.randperm(len(change_indices)).tolist()
        change_indices = [change_indices[i] for i in perm]

        mixed: List[int] = []
        max_len = max(len(all_indices), len(change_indices))
        for i in range(max_len):
            if i < len(change_indices):
                mixed.append(change_indices[i])
            if i < len(all_indices):
                mixed.append(all_indices[i])
        return iter(mixed[: self.dataset_size])

    def __len__(self) -> int:
        return self.dataset_size
