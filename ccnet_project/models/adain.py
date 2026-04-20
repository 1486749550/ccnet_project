import torch
import torch.nn as nn


class AdaIN2d(nn.Module):
    """
    二维自适应实例归一化。

    公式含义：
    1. 先对 content 的每个通道做实例归一化，消除其原有均值和方差；
    2. 再使用 style 提供的通道统计量重新缩放和平移；
    3. 这样输出特征保留了 content 的空间结构，但具有 style 的全局风格。
    """

    def __init__(self, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, content: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
        assert content.shape[0] == style.shape[0], "AdaIN 中 content 与 style 的 batch 大小必须一致"
        assert content.shape[1] == style.shape[1], "AdaIN 中 content 与 style 的通道数必须一致"
        assert content.ndim == 4, f"AdaIN content 必须为四维张量，当前为: {content.shape}"
        assert style.ndim == 4 and style.shape[-2:] == (1, 1), f"AdaIN style 必须为 [B,C,1,1]，当前为: {style.shape}"

        content_mean = content.mean(dim=(2, 3), keepdim=True)
        content_var = content.var(dim=(2, 3), keepdim=True, unbiased=False)
        content_std = torch.sqrt(content_var + self.eps)
        normalized = (content - content_mean) / content_std

        style_mean = style
        style_std = style.abs() + 1.0
        out = normalized * style_std + style_mean
        assert out.shape == content.shape, f"AdaIN 输出尺寸必须与 content 一致，当前为: {out.shape} vs {content.shape}"
        return out
