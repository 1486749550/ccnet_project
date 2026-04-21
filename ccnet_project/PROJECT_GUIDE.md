# CCNet Project 使用指南与文件说明

## 1. 项目简介

本项目是一个基于 **PyTorch** 的 **CCNet（Content Cleansing Network）** 复现工程，目标任务是双时相遥感图像变化检测中的**伪变化抑制**。

核心思想可以概括为三句话：

1. 对双时相图像分别提取两类表示：`content`（地物内容）与 `style`（成像风格）。
2. 用 `content` 分支完成变化检测，尽量避免把光照、色偏、季节等风格差异误判为变化。
3. 用 `content + style` 恢复原图，通过恢复任务反向约束网络完成更合理的内容/风格解耦。

当前工程已经包含：

- 可运行的模型实现
- 数据集与同步增强
- 四类损失与总损失
- 训练 / 验证 / 推理入口
- 混淆矩阵指标统计
- 伪数据集生成脚本
- 最小 smoke test

---

## 2. 项目目录结构

```text
ccnet_project/
├── configs/
│   └── ccnet_base.yaml
├── datasets/
│   ├── dataset.py
│   ├── transforms.py
│   └── sampler.py
├── models/
│   ├── blocks.py
│   ├── mrb.py
│   ├── style_encoder.py
│   ├── adain.py
│   ├── change_decoder.py
│   ├── restoration_decoder.py
│   └── ccnet.py
├── losses/
│   ├── change_loss.py
│   ├── separation_loss.py
│   ├── similarity_loss.py
│   ├── restoration_loss.py
│   └── total_loss.py
├── metrics/
│   └── seg_metrics.py
├── engine/
│   ├── trainer.py
│   ├── evaluator.py
│   └── inferencer.py
├── utils/
│   ├── logger.py
│   ├── seed.py
│   ├── checkpoint.py
│   └── misc.py
├── generate_dummy_dataset.py
├── train.py
├── validate.py
├── infer.py
└── PROJECT_GUIDE.md
```

---

## 3. 运行前准备

本工程默认你已经有可用的 Python + PyTorch 环境，不需要额外创建虚拟环境。

如果只是想快速验证工程流程，推荐直接使用项目内的伪数据集生成脚本。

### 3.1 推荐的最小验证流程

```bash
python generate_dummy_dataset.py
python train.py --config configs/ccnet_base.yaml
python validate.py --config configs/ccnet_base.yaml --checkpoint outputs/ccnet_base/best_f1.pth
python infer.py --config configs/ccnet_base.yaml --checkpoint outputs/ccnet_base/best_f1.pth --save_dir outputs/infer
```

### 3.2 当前默认配置说明

配置文件位于 [configs/ccnet_base.yaml](/c:/Users/Administrator/Desktop/课题组/mask_cd/ccnet_project/configs/ccnet_base.yaml:1)。

主要配置项如下：

- `seed`：随机种子
- `device`：优先使用 `cuda`
- `dataset.train_root / val_root / test_root`：训练、验证、测试数据目录
- `dataset.image_size`：输入图像统一缩放尺寸，要求能被 16 整除
- `model.base_channels`：主干网络基础通道数，默认 32
- `loss.lambda_sep / lambda_sim / lambda_rst`：三项辅助损失权重
- `optim.epochs / batch_size / lr`：训练轮数、批大小、学习率
- `optim.amp`：是否启用自动混合精度
- `inference.threshold`：推理时的二值化阈值

---

## 4. 数据集组织格式

真实数据或伪数据都需要遵循同一个目录格式：

```text
data/
├── train/
│   ├── t1/
│   ├── t2/
│   └── mask/
├── val/
│   ├── t1/
│   ├── t2/
│   └── mask/
└── test/
    ├── t1/
    ├── t2/
    └── mask/
```

其中同名文件必须一一对应，例如：

```text
train/t1/0001.png
train/t2/0001.png
train/mask/0001.png
```

说明：

- `t1/`：时相 1 图像
- `t2/`：时相 2 图像
- `mask/`：变化标签，取值为 0 或 1

---

## 5. 使用伪数据集快速跑通工程

脚本文件为 [generate_dummy_dataset.py](/c:/Users/Administrator/Desktop/课题组/mask_cd/ccnet_project/generate_dummy_dataset.py:1)。

### 5.1 最简单的使用方式

```bash
python generate_dummy_dataset.py
```

默认会生成：

- `data/train`：16 个样本
- `data/val`：4 个样本
- `data/test`：4 个样本

### 5.2 自定义参数

```bash
python generate_dummy_dataset.py --root data --image_size 256 --train_samples 32 --val_samples 8 --test_samples 8 --seed 42
```

参数说明：

- `--root`：输出根目录，默认 `data`
- `--image_size`：图像尺寸，必须能被 16 整除
- `--train_samples`：训练样本数
- `--val_samples`：验证样本数
- `--test_samples`：测试样本数
- `--seed`：随机种子

### 5.3 伪数据生成逻辑

该脚本会生成两类变化来源：

- 真实变化：在 `t2` 上随机绘制矩形或圆形区域，并同步写入 `mask`
- 风格扰动：对整图施加亮度、对比度、色偏、轻雾等变换，但**不改变 mask**

这样做的意义是：

- 真实变化用于训练网络学习“地物真的变了”
- 风格扰动用于模拟“看起来变了但其实没变”的伪变化来源

这与 CCNet 的问题设定是匹配的。

---

## 6. 训练、验证、推理指南

## 6.1 训练

入口文件是 [train.py](/c:/Users/Administrator/Desktop/课题组/mask_cd/ccnet_project/train.py:1)。

运行命令：

```bash
python train.py --config configs/ccnet_base.yaml
```

训练阶段会做这些事情：

1. 读取配置文件
2. 设置随机种子
3. 构建训练集与验证集 DataLoader
4. 创建 CCNet 模型
5. 创建 Trainer
6. 执行标准 epoch 训练循环
7. 在每个 epoch 后做验证
8. 保存 checkpoint

训练日志会输出：

- `loss_total`
- `loss_cd`
- `loss_sep`
- `loss_sim`
- `loss_rst`
- 验证集 `f1 / iou / accuracy`

模型权重默认保存在：

```text
outputs/ccnet_base/latest.pth
outputs/ccnet_base/best_f1.pth
```

## 6.2 验证

入口文件是 [validate.py](/c:/Users/Administrator/Desktop/课题组/mask_cd/ccnet_project/validate.py:1)。

运行命令：

```bash
python validate.py --config configs/ccnet_base.yaml --checkpoint outputs/ccnet_base/best_f1.pth
```

验证输出指标包括：

- `precision`
- `recall`
- `f1`
- `accuracy`
- `iou`
- `tp`
- `tn`
- `fp`
- `fn`
- `loss_total`

注意：这些指标是**基于整个验证集累计混淆矩阵**计算的，不是逐图平均。

## 6.3 推理

入口文件是 [infer.py](/c:/Users/Administrator/Desktop/课题组/mask_cd/ccnet_project/infer.py:1)。

运行命令：

```bash
python infer.py --config configs/ccnet_base.yaml --checkpoint outputs/ccnet_base/best_f1.pth --save_dir outputs/infer
```

推理输出目录：

```text
outputs/infer/
├── prob/
└── pred/
```

说明：

- `prob/`：变化类别概率图
- `pred/`：最终二值变化图

---

## 7. 模型整体流程说明

输入为两张双时相图像：

- `x_t1: [B, 3, H, W]`
- `x_t2: [B, 3, H, W]`

其中要求：

- 两个输入必须四维
- 两者形状必须完全一致
- `H` 和 `W` 必须能被 16 整除

整体前向流程如下：

1. `x_t1` 和 `x_t2` 分别通过共享权重 `MRB` 内容骨干，得到 5 级内容特征
2. `x_t1` 和 `x_t2` 分别通过独立 `StyleEncoder`，得到两个风格编码
3. `ChangeDecoder` 对双时相内容特征进行逐级融合，输出 2 通道变化 logits
4. `softmax + argmax` 得到最终变化预测 `pred_mask`
5. 共享权重 `RestorationDecoder` 分别用 `content + style` 恢复 `t1` 和 `t2`
6. 若训练阶段提供 `target_mask`，则继续计算总损失

最终 `forward` 返回字典中至少包含：

- `change_logits`
- `pred_mask`
- `content_t1`
- `content_t2`
- `style_t1`
- `style_t2`
- `rst_t1`
- `rst_t2`

---

## 8. 损失函数设计说明

总损失定义为：

```text
L = L_cd + λ1 * L_sep + λ2 * L_sim + λ3 * L_rst
```

默认权重为：

- `λ1 = 0.1`
- `λ2 = 0.1`
- `λ3 = 0.1`

### 8.1 变化检测损失 `L_cd`

使用：

- 2 通道 `change_logits`
- `target_mask` 生成的 one-hot 标签
- `BCEWithLogitsLoss`

作用：

- 直接监督变化检测主任务

### 8.2 特征分离损失 `L_sep`

使用：

- 最深层内容特征 `c5`
- 风格编码 `style`

当前工程采取的是**工程化近似方案**：

1. 对 `c5` 做全局平均池化得到 `[B, C]`
2. 将 `style` reshape 为 `[B, C]`
3. 计算二者的相关矩阵
4. 最小化其 Frobenius norm 的平方

作用：

- 抑制 style 信息泄漏到 content 表示中

### 8.3 内容相似性损失 `L_sim`

作用区域：

- 只在 `unchanged_mask = 1 - target_mask` 的区域中计算

当前工程采取的是**工程化近似方案**：

1. 将未变化区域 mask 下采样到各个尺度
2. 在各尺度内容特征上提取未变化区域响应
3. 使用排序后的特征差值近似 `Sliced Wasserstein Distance`
4. 对多尺度结果求平均

这样做的动机是：

- 未变化区域的 content 在两个时相中应该尽量一致
- 但直接做像素级强约束通常不够稳定

### 8.4 图像恢复损失 `L_rst`

使用：

- `rst_t1` 与 `x_t1` 的 L1 损失
- `rst_t2` 与 `x_t2` 的 L1 损失
- 最终取平均

作用：

- 迫使网络保留足够的信息完成图像重建
- 防止所谓“解耦”变成简单丢信息

---

## 9. Smoke Test 使用说明

文件 [models/ccnet.py](/c:/Users/Administrator/Desktop/课题组/mask_cd/ccnet_project/models/ccnet.py:1) 中带有最小 smoke test。

运行命令：

```bash
python models/ccnet.py
```

该测试会：

1. 随机生成 `x_t1`
2. 随机生成 `x_t2`
3. 随机生成 `mask`
4. 跑通 `forward`
5. 跑通 `loss`
6. 跑通 `metrics`
7. 打印所有关键输出张量 shape
8. 用 `assert` 检查结果是否符合预期

适合在以下场景使用：

- 你刚改完模型结构，想先看前向是否还能跑
- 你怀疑 loss 或 shape 对不齐
- 你还没准备真实数据，但要先确认工程可运行

---

## 10. 每个 Python 文件的解释说明

下面按文件逐一说明其职责与阅读重点。

## 10.1 根目录文件

### `train.py`

作用：

- 训练入口脚本

主要工作：

- 加载 yaml 配置
- 构建训练集和验证集
- 初始化模型
- 初始化 Trainer
- 启动训练

什么时候看它：

- 想改训练入口参数
- 想替换数据路径
- 想调整训练启动逻辑

### `validate.py`

作用：

- 验证入口脚本

主要工作：

- 加载配置和 checkpoint
- 构建验证集 DataLoader
- 调用 Evaluator 做整体验证

什么时候看它：

- 想离线评估某个权重文件
- 想增加额外验证输出

### `infer.py`

作用：

- 推理入口脚本

主要工作：

- 加载测试集
- 加载模型权重
- 调用 Inferencer 保存概率图和预测图

什么时候看它：

- 想做批量推理
- 想修改输出图保存格式

### `generate_dummy_dataset.py`

作用：

- 生成可直接训练的伪双时相变化检测数据

主要工作：

- 生成背景纹理
- 合成真实变化区域
- 添加整图风格扰动
- 保存 `t1/t2/mask`

什么时候看它：

- 没有现成数据，但想先跑通训练流程
- 想快速做功能调试

---

## 10.2 `datasets/` 目录

### `datasets/dataset.py`

作用：

- 定义双时相变化检测数据集类 `BitemporalChangeDataset`

主要返回格式：

```python
{
    "img_t1": Tensor,   # [3, H, W]
    "img_t2": Tensor,   # [3, H, W]
    "mask": Tensor,     # [H, W]
    "name": str
}
```

主要工作：

- 扫描 `t1/t2/mask` 目录
- 检查同名样本是否完整
- 读取 RGB 图像与灰度标签
- 调用同步增强模块

阅读重点：

- `_scan_samples`
- `__getitem__`

### `datasets/transforms.py`

作用：

- 定义双时相同步增强与归一化逻辑

主要增强：

- resize
- flip
- rotate90
- shift
- normalize

关键特点：

- 所有几何变换都对 `img_t1 / img_t2 / mask` 严格同步

阅读重点：

- `PairedTransform`
- `_random_shift`
- `__call__`

### `datasets/sampler.py`

作用：

- 提供一个示例采样器 `AlternatingSubsetSampler`

当前状态：

- 训练主流程中默认没有启用

适用场景：

- 如果后续你想做类别不均衡采样，可以在这里扩展

---

## 10.3 `models/` 目录

### `models/blocks.py`

作用：

- 存放基础网络块

包含内容：

- `ConvBNReLU`
- `ResidualConvBlock`
- `DownsampleBlock`
- `UpsampleFuseBlock`

意义：

- 这些是多个模块公用的基础积木

### `models/mrb.py`

作用：

- 实现内容分支主干 `MaintainResolutionBackbone`

结构特点：

- 五个并行分支
- 五个尺度同时维护
- 残差块提特征
- 跨分支多尺度融合

输出：

- `c1 ~ c5`

阅读重点：

- `_init_branches`
- `_fuse_once`
- `forward`

### `models/style_encoder.py`

作用：

- 实现风格编码器 `StyleEncoder`

输出：

- `style: [B, 512, 1, 1]`

特点：

- 逐步降低空间分辨率
- 最终只保留全局风格统计

### `models/adain.py`

作用：

- 实现 `AdaIN2d`

功能：

- 先对 content 做实例归一化
- 再用 style 进行通道级重缩放和偏移

意义：

- 保留内容结构
- 注入风格属性

### `models/change_decoder.py`

作用：

- 实现变化检测解码器 `ChangeDecoder`

结构：

- `d5_cd = bottleneck(concat(c5_t1, c5_t2))`
- 逐级上采样并融合双时相特征
- 输出两通道变化 logits

输出：

- `change_logits: [B, 2, H, W]`
- `pred_mask: [B, H, W]`

### `models/restoration_decoder.py`

作用：

- 实现共享权重的图像恢复解码器

结构特点：

- 最深层用 `content + style` 初始化
- 每一级执行：
  上采样 -> 拼接当前尺度 content -> 卷积融合 -> AdaIN 注入 style

输出：

- `rst_t1`
- `rst_t2`

### `models/ccnet.py`

作用：

- 整体模型封装

主要内容：

- 组装内容分支、风格分支、变化解码器、恢复解码器
- 定义统一 `forward`
- 进行关键 assert 检查
- 提供 smoke test

这是最值得优先阅读的模型文件。

---

## 10.4 `losses/` 目录

### `losses/change_loss.py`

作用：

- 实现变化检测损失 `ChangeDetectionLoss`

特点：

- 使用 `BCEWithLogitsLoss`
- 在内部把 `target_mask` 转成 one-hot

### `losses/separation_loss.py`

作用：

- 实现特征分离损失 `FeatureSeparationLoss`

特点：

- 对 `c5` 和 `style` 的相关性做约束
- 使用工程化近似方式实现论文思想

### `losses/similarity_loss.py`

作用：

- 实现内容相似性损失 `ContentSimilarityLoss`

特点：

- 只在未变化区域计算
- 多尺度近似 Sliced Wasserstein

### `losses/restoration_loss.py`

作用：

- 实现恢复损失 `RestorationLoss`

特点：

- 对两个时相分别做 L1 再平均

### `losses/total_loss.py`

作用：

- 统一汇总全部损失

输出：

- `loss_total`
- `loss_cd`
- `loss_sep`
- `loss_sim`
- `loss_rst`

这是训练阶段最重要的损失总入口。

---

## 10.5 `metrics/` 目录

### `metrics/seg_metrics.py`

作用：

- 实现基于累计混淆矩阵的指标器 `ConfusionMatrixMeter`

支持方法：

- `reset()`
- `update(pred, target)`
- `compute()`

输出指标：

- `precision`
- `recall`
- `f1`
- `accuracy`
- `iou`
- `tp`
- `tn`
- `fp`
- `fn`

---

## 10.6 `engine/` 目录

### `engine/trainer.py`

作用：

- 封装训练主流程

主要内容：

- 优化器初始化
- 学习率调度器初始化
- AMP 控制
- 单个 epoch 的训练
- 验证调用
- checkpoint 保存

阅读重点：

- `train`
- `train_one_epoch`

### `engine/evaluator.py`

作用：

- 封装验证流程

主要内容：

- 关闭梯度
- 跑验证集前向
- 计算总损失
- 累计混淆矩阵
- 汇总最终指标

### `engine/inferencer.py`

作用：

- 封装推理流程

主要内容：

- 前向推理
- 保存概率图
- 保存二值预测图

---

## 10.7 `utils/` 目录

### `utils/logger.py`

作用：

- 创建日志器

功能：

- 同时输出到控制台和日志文件

### `utils/seed.py`

作用：

- 设置随机种子

意义：

- 提高实验可复现性

### `utils/checkpoint.py`

作用：

- 保存和加载 checkpoint

包含：

- `save_checkpoint`
- `load_checkpoint`

### `utils/misc.py`

作用：

- 放一些通用小工具

包含：

- `AverageMeter`
- `load_yaml`
- `ensure_dir`

---

## 11. 关键张量说明

训练和调试时最常用的关键张量如下。

### 11.1 输入张量

- `img_t1: [B, 3, H, W]`
- `img_t2: [B, 3, H, W]`
- `mask: [B, H, W]`

### 11.2 内容特征

- `c1: [B, 32, H, W]`
- `c2: [B, 64, H/2, W/2]`
- `c3: [B, 128, H/4, W/4]`
- `c4: [B, 256, H/8, W/8]`
- `c5: [B, 512, H/16, W/16]`

### 11.3 风格编码

- `style_t1: [B, 512, 1, 1]`
- `style_t2: [B, 512, 1, 1]`

### 11.4 变化检测输出

- `change_logits: [B, 2, H, W]`
- `pred_mask: [B, H, W]`

### 11.5 恢复输出

- `rst_t1: [B, 3, H, W]`
- `rst_t2: [B, 3, H, W]`

---

## 12. 常见问题与排查建议

### 12.1 报错：输入尺寸不能被 16 整除

原因：

- MRB 主干包含 5 级尺度，最深层是 `H/16, W/16`

解决：

- 把配置中的 `dataset.image_size` 改成 256、512 这类能被 16 整除的值

### 12.2 报错：找不到样本文件

原因：

- `t1/t2/mask` 文件名不一致

解决：

- 检查同名样本是否完整存在

### 12.3 训练能跑，但 F1 很低

可能原因：

- 伪数据集过小
- 学习轮数太少
- 学习率不合适
- 数据分布与真实任务差距较大

建议：

- 先增大 `train_samples`
- 增加 `epochs`
- 在真实数据上重新调参

### 12.4 想快速检查模型结构是否正常

直接运行：

```bash
python models/ccnet.py
```

这通常比直接开训练更适合排 shape 错误。

---

## 13. 推荐阅读顺序

如果你是第一次接触这个工程，建议按下面顺序读代码：

1. `models/ccnet.py`
2. `models/mrb.py`
3. `models/change_decoder.py`
4. `models/restoration_decoder.py`
5. `losses/total_loss.py`
6. `datasets/dataset.py`
7. `datasets/transforms.py`
8. `engine/trainer.py`
9. `train.py`

这样比较容易先建立整体图景，再看细节。

---

## 14. 一套最常用命令汇总

### 14.1 生成伪数据

```bash
python generate_dummy_dataset.py
```

### 14.2 训练

```bash
python train.py --config configs/ccnet_base.yaml
```

### 14.3 验证

```bash
python validate.py --config configs/ccnet_base.yaml --checkpoint outputs/ccnet_base/best_f1.pth
```

### 14.4 推理

```bash
python infer.py --config configs/ccnet_base.yaml --checkpoint outputs/ccnet_base/best_f1.pth --save_dir outputs/infer
```

### 14.5 Smoke test

```bash
python models/ccnet.py
```

---

## 15. 后续可扩展方向

如果你准备把这个工程进一步用于实验或论文复现，比较值得继续扩展的点有：

- 增加真实遥感数据集适配器
- 增加多卡训练支持
- 增加 tensorboard 或 wandb 日志
- 增加更严格的论文版 MRB 结构细节
- 增加更多恢复损失或对比学习损失
- 增加测试时翻转增强（TTA）
- 增加可视化脚本，直接对比 `t1 / t2 / pred / gt`

---

这份文档对应的是当前工作区内的实际实现版本。如果后续你继续修改模型结构，建议同步更新本文件，避免“代码和说明脱节”。

---

## 16. 继续训练与训练集划分验证集

### 16.1 继续训练

继续训练入口文件是 `continue_train.py`。它会从已有 checkpoint 中加载模型权重，并在默认情况下恢复 optimizer、scheduler、epoch 和 `best_f1`。

基础用法：

```bash
python continue_train.py --config configs/ccnet_base.yaml --checkpoint outputs/ccnet_base/latest.pth --epochs 51
```

常用参数：

- `--checkpoint`：要加载的 checkpoint，例如 `outputs/ccnet_base/latest.pth` 或 `outputs/ccnet_base/best_f1.pth`
- `--epochs`：继续训练后的总 epoch 数，不是额外追加轮数
- `--lr`：覆盖配置文件和 checkpoint 中的学习率
- `--batch_size`：覆盖配置文件中的 batch size
- `--save_dir`：覆盖 checkpoint 输出目录
- `--weights_only`：只加载模型权重，重新初始化 optimizer 和 scheduler

示例：

```bash
python continue_train.py --config configs/ccnet_base.yaml --checkpoint outputs/ccnet_base/latest.pth --epochs 80 --lr 0.0001 --save_dir outputs/continue_train
```

只加载模型权重并重新初始化训练状态：

```bash
python continue_train.py --config configs/ccnet_base.yaml --checkpoint outputs/ccnet_base/best_f1.pth --weights_only --epochs 50 --lr 0.0001
```

### 16.2 新数据集没有验证集时的处理

项目支持从训练集中自动划分一部分样本作为验证集。配置项位于 `dataset` 下：

```yaml
dataset:
  train_root: data/train
  val_root: data/val
  use_train_val_split: false
  val_split: 0.2
```

规则：

- 如果 `val_root` 存在，并且 `use_train_val_split: false`，会使用独立验证集。
- 如果 `val_root` 不存在，会自动从 `train_root` 中划分验证集。
- 如果设置 `use_train_val_split: true`，即使 `val_root` 存在，也会强制从 `train_root` 中划分验证集。
- `val_split: 0.2` 表示 20% 样本作为验证集，80% 样本用于训练。
- 划分使用配置中的 `seed`，因此同一个 seed 下划分结果可复现。

继续训练时也可以直接通过命令行指定：

```bash
python continue_train.py --config configs/ccnet_base.yaml --checkpoint outputs/ccnet_base/latest.pth --use_train_val_split --val_split 0.2
```

如果你的新数据集只有训练目录，例如：

```text
new_data/train/
├── t1/
├── t2/
└── mask/
```

可以在 config 中这样设置：

```yaml
dataset:
  train_root: new_data/train
  val_root: new_data/val
  use_train_val_split: true
  val_split: 0.2
```

然后运行：

```bash
python continue_train.py --config configs/ccnet_base.yaml --checkpoint outputs/ccnet_base/latest.pth --epochs 80 --lr 0.0001
```

### 16.3 更新后的常用命令

从零训练：

```bash
python train.py --config configs/ccnet_base.yaml
```

继续训练：

```bash
python continue_train.py --config configs/ccnet_base.yaml --checkpoint outputs/ccnet_base/latest.pth --epochs 80
```

继续训练并从训练集划分验证集：

```bash
python continue_train.py --config configs/ccnet_base.yaml --checkpoint outputs/ccnet_base/latest.pth --epochs 80 --use_train_val_split --val_split 0.2
```
