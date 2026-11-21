# Hybrid Muon

[English](README.md) · 简体中文

这个 PyTorch 优化器（optimizer）按参数分组：选定的投影矩阵（projection matrices）使用 Muon，其余可训练参数（trainable parameters）使用 AdamW。

Python 3.10+ · PyTorch 2.10+ · [包版本](src/hybrid_muon/__init__.py)

[快速开始](#快速开始) · [参数分组](#参数分组) · [接入训练框架](#接入训练框架) · [性能](#性能) · [Qwen 模型指南（英文）](docs/qwen-models.md)

## 快速开始

**克隆仓库并以可编辑模式安装**

```sh
git clone https://github.com/JamePeng/hybrid_muon.git
cd hybrid_muon
python -m pip install -e .
```

**直接从 GitHub 安装**

```sh
python -m pip install "git+https://github.com/JamePeng/hybrid_muon.git"
```

安装后，运行 `python -c "import hybrid_muon; print(hybrid_muon.__version__)"` 查看版本号。

先挂载可选模块，设置哪些权重参与训练，并将模型移到训练设备，再创建优化器。下面的示例显式写出了默认学习率（learning rate，LR）和权重衰减（weight decay），并打印实际分组。

```python
from hybrid_muon import HybridMuonAdamW

optimizer = HybridMuonAdamW(
    model,
    muon_config={"lr": 5e-5, "weight_decay": 0.005},
    adamw_lr=1.5e-5,
    adamw_weight_decay=0.01,
)
for group in optimizer.param_groups:
    print(group["name"], group["lr"], group["weight_decay"], group["param_names"])
```

在 PyTorch 训练循环中，先调用 `optimizer.zero_grad()` 清空梯度，完成前向传播（forward pass）和反向传播（backward pass）后，再调用 `optimizer.step()` 更新参数。

## 参数分组

创建优化器之前可以先打印分组报告。即使没有投影权重匹配 Muon，也能查看原因。

```python
from hybrid_muon import inspect_model

print(inspect_model(model))  # Set verbose=True to list all grouped tensors.
```

要预览本地检查点（checkpoint），先安装 `safetensors`，再运行 `python -m hybrid_muon inspect "/path/to/model"`。加上 `--verbose` 可列出所有张量的名称和形状。命令行工具只读取张量头部信息，按所有已存张量均可训练来统计。挂载可选模块或冻结层之后，用模型对象调用 `inspect_model` 即可查看实际分组。两种报告都复用优化器的分组规则，并显示默认超参数。

注意力层和多层感知机（multilayer perceptron，MLP）的投影矩阵负责混合特征。Muon 对动量更新（momentum update）做近似正交化（orthogonalization），减弱最大奇异值对应方向对更新的支配作用。AdamW 则按元素调整步长，让词嵌入、归一化缩放参数和偏置根据各自的梯度统计量更新。Muon 和 AdamW 的学习率可以独立设置。[Muon 原始设计](https://kellerjordan.github.io/posts/muon/)

| 层或参数（Layer or parameter） | 权重名或示例（Weight names or examples） | 参数组（Parameter group） | 默认学习率（Default LR） | 权重衰减（Weight decay） |
| --- | --- | --- | ---: | ---: |
| 自注意力投影（Self-attention projections） | `self_attn.{q_proj,k_proj,v_proj,o_proj}.weight` | `muon` | 5e-5 | 0.005 |
| MLP 投影（MLP projections） | `mlp.{gate_proj,up_proj,down_proj}.weight` | `muon` | 5e-5 | 0.005 |
| 线性注意力投影（Linear-attention projections） | `linear_attn.{in_proj_qkv,in_proj_z,out_proj}.weight` | `muon` | 5e-5 | 0.005 |
| 词嵌入和输出头（Token embeddings and output head） | `embed_tokens.weight`、`lm_head.weight` | `adamw_no_decay` | 1.5e-5 | 0 |
| 归一化参数、偏置、标量和向量（Normalization, biases, scalars, and vectors） | `self_attn.q_norm.weight`、`mlp.up_proj.bias` | `adamw_no_decay` | 1.5e-5 | 0 |
| 其他可训练权重（Other trainable weights） | `linear_attn.in_proj_a.weight`、`linear_attn.in_proj_b.weight`、`linear_attn.conv1d.weight`、`mtp.fc.weight` | `adamw_decay` | 1.5e-5 | 0.01 |

构造优化器时只遍历一次 `model.named_parameters()`，跳过冻结参数（frozen parameters），共享权重（tied weights）只计一次。进入 Muon 组需要同时满足三个条件：权重是**二维张量**，名称包含 `.layers.`，且后缀匹配表中的投影权重。剩余参数中，标量、一维张量，以及名称包含 `norm`、`bias`、`embed_tokens` 或 `lm_head` 的张量使用零衰减 AdamW，其余使用带权重衰减的 AdamW。

以 Qwen3.5 的多词元预测（multi-token prediction，MTP）头为例，`mtp.layers.0.self_attn.q_proj.weight` 使用 Muon，`mtp.fc.weight` 使用带衰减的 AdamW，`mtp.norm.weight` 使用零衰减 AdamW。请将已挂载 MTP 头的模型传入优化器。适配其他架构时，可用快速开始示例中的 `param_names` 核对分组。即使使用适配器（adapters），也至少要有一个匹配的投影权重参与训练，才能成功构造优化器。[Qwen3、Qwen3.5 和 Qwen3.8-27B 的层结构与预期分组数量（英文）](docs/qwen-models.md)

## 接入训练框架

| 训练方式 | 接入方法 |
| --- | --- |
| TRL `SFTTrainer`，单卡或普通 DDP | 传入 `optimizers=(optimizer, None)`，由 Trainer 创建学习率调度器（learning rate scheduler）。 |
| PyTorch 分布式数据并行（DistributedDataParallel，DDP） | 用 `ddp_model.module` 创建优化器，通过 `ddp_model` 做前向传播。每个进程持有完整的参数矩阵和优化器状态（optimizer state）。 |
| DeepSpeed stage 0 | 将优化器传给引擎。参数保持 FP32，通过自动类型转换（autocast）对适用的运算使用 BF16。 |
| DeepSpeed ZeRO stage 1–3 | 标准包装器向底层优化器传入展平后的参数分片（flat parameter partitions），而本包的 `step()` 需要完整的二维投影矩阵。stage 3 还会对模型权重分片。[ZeRO 各阶段说明](https://deepspeed.readthedocs.io/en/stable/zero3.html) |

[JamePeng](https://github.com/JamePeng) 曾在独立训练脚本中使用 ZeRO-2 完成 Qwen3 监督微调（supervised fine-tuning，SFT），该脚本有自己的优化器实现。DeepSpeed 也为其内置的 `MuonWithAuxAdam` 实现了保留矩阵边界的 ZeRO 更新路径。对于传入的自定义优化器，启用 `zero_allow_untested_optimizer` 只会改变 DeepSpeed 的优化器检查。[DeepSpeed ZeRO 实现](https://github.com/deepspeedai/DeepSpeed/blob/b5e0ba67f2757af1e2003fba9dc362db042a5ffe/deepspeed/runtime/zero/stage_1_and_2.py)

<details>
<summary><strong>TRL SFTTrainer</strong> · BF16 与分组学习率</summary>

将快速开始中创建的优化器与模型、分词器（tokenizer）、训练集一起传给 Trainer。Trainer 负责梯度累积（gradient accumulation）和参数更新，线性调度器从各参数组自己的初始学习率开始调整。[TRL SFTTrainer 文档](https://huggingface.co/docs/trl/sft_trainer)

```python
from trl import SFTConfig, SFTTrainer

args = SFTConfig(output_dir="checkpoints", bf16=True, lr_scheduler_type="linear")
trainer = SFTTrainer(
    model=model,
    args=args,
    processing_class=tokenizer,
    train_dataset=train_dataset,
    optimizers=(optimizer, None),
)
trainer.train()
```

普通 DDP 为每张 GPU 启动一个进程。例如，两张卡可以运行 `torchrun --standalone --nproc-per-node=2 train_sft.py`。TRL 与 DeepSpeed 一起使用时，参见下方的 stage 0 示例。

</details>

<details>
<summary><strong>PyTorch DDP</strong> · 训练步骤与检查点</summary>

初始化进程组（process group），将模型放到当前进程对应的 GPU 上，再用 DDP 包装。使用 `DistributedSampler` 为每个进程分配各自的批次。DDP 在反向传播时同步梯度。在支持 BF16 的 GPU 上，模型参数保持 FP32，前向传播启用自动混合精度。

```python
import torch
from hybrid_muon import HybridMuonAdamW

optimizer = HybridMuonAdamW(ddp_model.module)
optimizer.zero_grad(set_to_none=True)
with torch.autocast("cuda", dtype=torch.bfloat16):
    loss = ddp_model(**batch).loss
loss.backward()
optimizer.step()
```

保存检查点（checkpoint）时，由一个进程写入 `ddp_model.module.state_dict()` 和 `optimizer.state_dict()`。恢复时，在每个进程上按相同的可训练参数名称和分组顺序重建模型与优化器，再加载这两份状态。仓库的双进程 CPU/Gloo 测试将 DDP 运行三步后的结果与单进程使用合并批次的结果进行了比较。[DDP 测试](tests/test_optimizer.py)

</details>

<details>
<summary><strong>DeepSpeed stage 0</strong> · 原生引擎与 TRL 配置</summary>

每张支持 BF16 的 GPU 对应一个进程。将未包装的 FP32 模型放到当前进程的 GPU 上。使用 Transformers 加载模型时，可给 `from_pretrained` 传入 `dtype=torch.float32`。下面的引擎和 TRL 示例共用这份配置。DeepSpeed 通过 `torch_autocast` 启用 BF16 计算，同时保留优化器所需的原始矩阵。配置中省略原生的 `bf16`、`fp16` 和 `optimizer` 条目。[DeepSpeed 混合精度文档](https://deepspeed.readthedocs.io/en/stable/training.html#mixed-precision-training)

```python
ds_config = {
    "train_micro_batch_size_per_gpu": 1,
    "zero_optimization": {"stage": 0},
    "torch_autocast": {"enabled": True, "dtype": "bfloat16"},
}
```

**使用原生引擎。** 传入快速开始中创建的优化器，由引擎执行反向传播和参数更新。

```python
import deepspeed

engine, _, _, _ = deepspeed.initialize(model=model, optimizer=optimizer, config=ds_config)
loss = engine(**batch).loss
engine.backward(loss)
engine.step()
```

保存检查点时，所有进程都要用同一个标签调用 `engine.save_checkpoint(directory, tag=tag)`。恢复时调用 `engine.load_checkpoint(directory, tag=tag)`。[检查点 API](https://deepspeed.readthedocs.io/en/stable/model-checkpointing.html)

**在 TRL 中使用 DeepSpeed。** 重写 `create_optimizer`，让 Trainer 在引擎初始化时构建参数组。批次配置由 Trainer 填充，混合精度由 DeepSpeed 管理。使用 Accelerate 启动时，选择 `--mixed_precision no`。

```python
from hybrid_muon import HybridMuonAdamW
from trl import SFTConfig, SFTTrainer


class HybridSFTTrainer(SFTTrainer):
    def create_optimizer(self, model=None):
        if self.optimizer is None:
            self.optimizer = HybridMuonAdamW(self.model if model is None else model)
        return self.optimizer


args = SFTConfig(
    output_dir="checkpoints",
    bf16=False,
    fp16=False,
    lr_scheduler_type="linear",
    deepspeed={
        **ds_config,
        "train_micro_batch_size_per_gpu": "auto",
        "gradient_accumulation_steps": "auto",
    },
)
trainer = HybridSFTTrainer(
    model=model, args=args, processing_class=tokenizer, train_dataset=train_dataset
)
trainer.train()
```

</details>

## 性能

**训练计算量。** Liu 等人在非嵌入参数（non-embedding parameters）数量为 3.99 亿至 15 亿的稠密模型上进行了预训练实验。根据计算最优缩放曲线（compute-optimal scaling curves）估算，达到相同损失时，其 Muon 方案所需的训练浮点运算量（training FLOPs）约为调优后 AdamW 的 **52%**。设置 `muon_config={"scale_mode": "paper"}` 可使用论文研究的按矩阵形状缩放更新量的方法。[Muon is Scalable for LLM Training](https://arxiv.org/html/2502.16982v1#S3.SS2)

**显存。** 在优化器状态精度相同的条件下，AdamW 为每个矩阵保存两个矩估计缓冲区（moment buffers），Muon 只保存一个动量缓冲区（momentum buffer），因此每个选中的矩阵可以少占用一个同尺寸的常驻缓冲区。Newton–Schulz 迭代本身还需要临时显存和额外的矩阵乘法。本仓库尚未测量 Qwen 的训练吞吐量（training throughput）和达到目标损失所需的时间。

### 循环 Transformer 实验

JamePeng 在研究循环 Transformer（loop transformers）时，使用一个 **3.09 亿参数的循环语言模型**（recurrent language model）比较了早期混合优化器实现与 AdamW。在矩阵学习率同为 `5e-5` 时，混合优化器的**峰值显存分配量减少了 207.6 MiB**。28 个选中矩阵共含 54,525,952 个参数，按少保存一个 FP32 缓冲区计算，预计可节省 **208.0 MiB**。

| 优化器 | 矩阵学习率 | 训练 NLL（128 个样本） | 验证 NLL（32 个新样本） | PyTorch 峰值显存分配量（peak allocated memory，GiB） |
| --- | ---: | ---: | ---: | ---: |
| AdamW | 5e-5 | 0.2312 | 6.0893 | 6.6365 |
| Hybrid (`paper`) | 2.5e-5 | 0.2575 | 5.6558 | 6.4337 |
| Hybrid (`paper`) | 3.5e-5 | 0.1638 | 5.7353 | 6.4337 |
| Hybrid (`paper`) | 5e-5 | 0.1131 | 5.8344 | 6.4337 |

四组实验从同一个模型检查点开始，分别初始化优化器状态，并在 RTX 5090 上按相同的批次顺序训练，使用 FP32 参数和 BF16 自动混合精度。每组运行 500 步，共处理 1,162,894 个参与损失计算的 token。28 个选中矩阵以外的参数统一使用 `1.5e-5` 的学习率。表中的负对数似然（negative log-likelihood，NLL）按 token 加权，数值越低越好。实验编号为 `optimizer-lr-validation-2026-09-26`。

<details>
<summary>使用 Muon 及其变体的模型报告</summary>

预训练中使用 Muon 或其变体的模型包括 [Kimi K2](https://arxiv.org/html/2507.20534v1#S2.SS1)、[Kimi K3](https://arxiv.org/html/2607.24653v1)、[GLM-4.5 和 GLM-5](https://arxiv.org/html/2602.15763v1)、[Step 3.5 Flash](https://arxiv.org/html/2602.10604v1)、[Inkling](https://thinkingmachines.ai/news/introducing-inkling/)、[Qwen3.8-Flash-Next](https://arxiv.org/html/2608.30320v1)、[DeepSeek-V4.1-Flash](https://arxiv.org/html/2609.19969v1) 和 [MAI Base 2](https://microsoft.ai/news/frontier-pre-training-at-microsoft-ai/)。小米在中期训练中使用了 Muon 的变体 Muown。[MiMo-V2.6 报告](https://arxiv.org/html/2610.11959v1)

</details>

## 作者与许可

JamePeng · <jame_peng@sina.com>

Hybrid Muon 采用 [Apache-2.0 许可证](LICENSE)。其中的 Newton–Schulz 迭代改编自 Keller Jordan 的 Muon 实现。再分发该部分代码的副本或实质性部分时，须保留其版权声明和许可声明。[上游源代码](https://github.com/KellerJordan/Muon/tree/f98f1cacc0263b04290753e32be8d498c1efc806) · [完整 MIT 声明](THIRD_PARTY_LICENSES.txt)

## 如何引用

如果你的项目或研究使用了 Hybrid Muon，请在 README、模型卡（model card）或论文中注明来源，并写明所用的发布版本或 Git 提交号，方便读者找到对应实现。

> JamePeng. *Hybrid Muon*. [https://github.com/JamePeng/hybrid_muon](https://github.com/JamePeng/hybrid_muon).
