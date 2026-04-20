import sys
from pathlib import Path
from typing import Dict, List

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn

from losses.total_loss import TotalLoss
from metrics.seg_metrics import ConfusionMatrixMeter
from models.change_decoder import ChangeDecoder
from models.mrb import MaintainResolutionBackbone
from models.restoration_decoder import RestorationDecoder
from models.style_encoder import StyleEncoder


class CCNet(nn.Module):
    """
    CCNet 主模型。

    结构说明：
    - 内容分支：共享权重 Siamese MRB
    - 风格分支：两个独立 StyleEncoder
    - 变化解码器：基于双时相 content 特征预测变化
    - 恢复解码器：共享权重，分别恢复 T1/T2 原图
    """

    def __init__(self, in_channels: int = 3, base_channels: int = 32) -> None:
        super().__init__()
        self.content_backbone = MaintainResolutionBackbone(in_channels=in_channels, base_channels=base_channels)
        self.style_encoder_t1 = StyleEncoder(in_channels=in_channels, base_channels=base_channels)
        self.style_encoder_t2 = StyleEncoder(in_channels=in_channels, base_channels=base_channels)
        self.change_decoder = ChangeDecoder(base_channels=base_channels)
        self.restoration_decoder = RestorationDecoder(base_channels=base_channels, out_channels=in_channels)

    def forward(
        self,
        x_t1: torch.Tensor,
        x_t2: torch.Tensor,
        target_mask: torch.Tensor | None = None,
        criterion: TotalLoss | None = None,
    ) -> Dict[str, torch.Tensor | List[torch.Tensor] | Dict[str, torch.Tensor]]:
        assert x_t1.ndim == 4, f"输入图像必须四维，x_t1 当前为: {x_t1.shape}"
        assert x_t2.ndim == 4, f"输入图像必须四维，x_t2 当前为: {x_t2.shape}"
        assert x_t1.shape == x_t2.shape, f"双时相输入形状必须相同，当前为: {x_t1.shape} vs {x_t2.shape}"
        h, w = x_t1.shape[-2:]
        assert h % 16 == 0 and w % 16 == 0, f"输入高宽必须能被 16 整除，当前为: {(h, w)}"

        content_t1 = self.content_backbone(x_t1)
        content_t2 = self.content_backbone(x_t2)
        style_t1 = self.style_encoder_t1(x_t1)
        style_t2 = self.style_encoder_t2(x_t2)

        c5_t1 = content_t1[-1]
        c5_t2 = content_t2[-1]
        assert style_t1.shape[1] == c5_t1.shape[1], "style 通道必须和最深层内容特征通道一致"
        assert style_t2.shape[1] == c5_t2.shape[1], "style 通道必须和最深层内容特征通道一致"

        change_logits, pred_mask = self.change_decoder(content_t1, content_t2)
        rst_t1 = self.restoration_decoder(content_t1, style_t1)
        rst_t2 = self.restoration_decoder(content_t2, style_t2)

        assert change_logits.shape[-2:] == (h, w), "最终输出 logits 高宽必须和输入一致"
        assert rst_t1.shape == x_t1.shape, "恢复图像尺寸必须与输入一致"
        assert rst_t2.shape == x_t2.shape, "恢复图像尺寸必须与输入一致"

        outputs: Dict[str, torch.Tensor | List[torch.Tensor] | Dict[str, torch.Tensor]] = {
            "change_logits": change_logits,
            "pred_mask": pred_mask,
            "content_t1": content_t1,
            "content_t2": content_t2,
            "style_t1": style_t1,
            "style_t2": style_t2,
            "rst_t1": rst_t1,
            "rst_t2": rst_t2,
        }

        if target_mask is not None and criterion is not None:
            outputs["losses"] = criterion(outputs, x_t1, x_t2, target_mask)
        return outputs


def _print_shape_dict(outputs: Dict[str, torch.Tensor | List[torch.Tensor] | Dict[str, torch.Tensor]]) -> None:
    """打印 smoke test 中各输出的形状。"""
    for key, value in outputs.items():
        if key == "losses":
            for loss_name, loss_value in value.items():
                print(f"{loss_name}: {float(loss_value.detach()):.6f}")
        elif isinstance(value, list):
            for idx, feat in enumerate(value, start=1):
                print(f"{key}[{idx}]: {tuple(feat.shape)}")
        elif torch.is_tensor(value):
            print(f"{key}: {tuple(value.shape)}")


if __name__ == "__main__":
    from losses.total_loss import TotalLoss

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CCNet(in_channels=3, base_channels=32).to(device)
    criterion = TotalLoss(lambda_sep=0.1, lambda_sim=0.1, lambda_rst=0.1).to(device)
    metric = ConfusionMatrixMeter()

    b, h, w = 2, 256, 256
    x_t1 = torch.randn(b, 3, h, w, device=device)
    x_t2 = torch.randn(b, 3, h, w, device=device)
    mask = torch.randint(0, 2, (b, h, w), device=device)

    outputs = model(x_t1, x_t2, target_mask=mask, criterion=criterion)
    _print_shape_dict(outputs)

    metric.update(outputs["pred_mask"], mask)
    scores = metric.compute()
    print("metrics:", scores)

    assert outputs["change_logits"].shape == (b, 2, h, w), "smoke test: change_logits 形状错误"
    assert outputs["pred_mask"].shape == (b, h, w), "smoke test: pred_mask 形状错误"
    assert outputs["rst_t1"].shape == x_t1.shape, "smoke test: rst_t1 形状错误"
    assert outputs["rst_t2"].shape == x_t2.shape, "smoke test: rst_t2 形状错误"
    print("smoke test passed")
