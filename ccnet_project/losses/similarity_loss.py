from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


class ContentSimilarityLoss(nn.Module):
    """
    内容相似性损失。

    这里采用工程化近似方案来替代严格的 Wasserstein/Sinkhorn：
    - 先在未变化区域内提取两个时相的内容特征
    - 再使用排序后的通道响应差值近似 Sliced Wasserstein Distance
    - 对多个尺度求平均

    该实现是可微且数值稳定的，适合作为论文思想的工程复现。
    """

    def __init__(self) -> None:
        super().__init__()

    def _masked_swd(self, feat_t1: torch.Tensor, feat_t2: torch.Tensor, unchanged_mask: torch.Tensor) -> torch.Tensor:
        assert feat_t1.shape == feat_t2.shape, "两个时相特征尺寸必须一致"
        resized_mask = F.interpolate(unchanged_mask.unsqueeze(1).float(), size=feat_t1.shape[-2:], mode="nearest")
        weighted_t1 = feat_t1 * resized_mask
        weighted_t2 = feat_t2 * resized_mask

        vec_t1 = weighted_t1.flatten(2)
        vec_t2 = weighted_t2.flatten(2)
        mask_flat = resized_mask.flatten(2)
        valid = mask_flat.sum(dim=2, keepdim=True).clamp_min(1.0)
        vec_t1 = vec_t1 / valid
        vec_t2 = vec_t2 / valid

        sorted_t1, _ = torch.sort(vec_t1, dim=2)
        sorted_t2, _ = torch.sort(vec_t2, dim=2)
        return torch.mean(torch.abs(sorted_t1 - sorted_t2))

    def forward(self, content_t1: List[torch.Tensor], content_t2: List[torch.Tensor], target_mask: torch.Tensor) -> torch.Tensor:
        assert len(content_t1) == len(content_t2), "content_t1 与 content_t2 的尺度数必须一致"
        unchanged_mask = 1 - target_mask.long()
        losses = []
        for feat_t1, feat_t2 in zip(content_t1, content_t2):
            losses.append(self._masked_swd(feat_t1, feat_t2, unchanged_mask))
        return torch.stack(losses).mean()
