# Hybrid Muon

English · [简体中文](README_zh.md)

A PyTorch optimizer that applies Muon to selected projection matrices and AdamW to the remaining trainable parameters.

Python 3.10+ · PyTorch 2.10+ · [Package version](src/hybrid_muon/__init__.py) · [Changelog](changelog.md)

[Quick start](#quick-start) · [Parameter groups](#parameter-groups) · [Training integrations](#training-integrations) · [Performance](#performance) · [Qwen model guide](docs/qwen-models.md)

## Quick start

**Install an editable clone**

```sh
git clone https://github.com/JamePeng/hybrid_muon.git
cd hybrid_muon
python -m pip install -e .
```

**Install directly from GitHub**

```sh
python -m pip install "git+https://github.com/JamePeng/hybrid_muon.git"
```

Print the installed version with `python -c "import hybrid_muon; print(hybrid_muon.__version__)"`.

Attach optional modules, set which weights are trainable, and move the model to its training device before creating the optimizer. This example sets the package defaults explicitly and shows how to inspect the assignments.

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

In a PyTorch loop, call `optimizer.zero_grad()`, run the forward and backward passes, then call `optimizer.step()`.

## Parameter groups

Preview the groups before building the optimizer, including when no projection matches:

```python
from hybrid_muon import inspect_model

print(inspect_model(model))  # Set verbose=True to list all grouped tensors.
```

To inspect a local checkpoint, install `safetensors` and run `python -m hybrid_muon inspect "/path/to/model"`. Add `--verbose` for all names and shapes. The CLI reads tensor headers and assumes every stored tensor is trainable. Use the model API after attaching optional heads or freezing layers to see the actual assignments. Both reports use the optimizer's grouping rules and display its default hyperparameters.

An attention or MLP projection mixes features through a matrix. Muon approximately orthogonalizes its momentum update, reducing the influence of the largest singular directions. AdamW adapts the step for each element, which lets embedding rows, normalization scales, and biases follow their own gradient statistics. The two optimizers have independent learning rates. [Original Muon design](https://kellerjordan.github.io/posts/muon/)

| Layer or parameter | Weight names or examples | Group | Default LR | Weight decay |
| --- | --- | --- | ---: | ---: |
| Self-attention projections | `self_attn.{q_proj,k_proj,v_proj,o_proj}.weight` | `muon` | 5e-5 | 0.005 |
| MLP projections | `mlp.{gate_proj,up_proj,down_proj}.weight` | `muon` | 5e-5 | 0.005 |
| Linear-attention projections | `linear_attn.{in_proj_qkv,in_proj_z,out_proj}.weight` | `muon` | 5e-5 | 0.005 |
| Token embeddings and output head | `embed_tokens.weight`, `lm_head.weight` | `adamw_no_decay` | 1.5e-5 | 0 |
| Normalization, biases, scalars, and vectors | `self_attn.q_norm.weight`, `mlp.up_proj.bias` | `adamw_no_decay` | 1.5e-5 | 0 |
| Other trainable weights | `linear_attn.in_proj_a.weight`, `linear_attn.in_proj_b.weight`, `linear_attn.conv1d.weight`, `mtp.fc.weight` | `adamw_decay` | 1.5e-5 | 0.01 |

The optimizer reads `model.named_parameters()` once, skips frozen parameters, and counts tied weights once. It assigns a tensor to Muon when it is **2D**, its name contains `.layers.`, and its suffix matches one of the projections above. Of the remaining tensors, those of rank 0 or 1 and those with `norm`, `bias`, `embed_tokens`, or `lm_head` in the name receive zero decay. All others receive AdamW weight decay.

For example, in a Qwen3.5 multi-token prediction (MTP) head, `mtp.layers.0.self_attn.q_proj.weight` matches Muon, `mtp.fc.weight` receives AdamW with decay, and `mtp.norm.weight` receives AdamW with zero decay. Pass the model that owns the attached head. For another architecture, inspect `param_names` in the quick start. Construction requires at least one trainable matching projection, including when using adapters. [Layer maps and expected group counts for Qwen3, Qwen3.5, and Qwen3.8-27B](docs/qwen-models.md)

### Qwen checkpoint inspection examples

These commands inspect three local checkpoints. Replace the paths with your model directories.

```powershell
python -m hybrid_muon inspect "/path/to/Qwen3-4B-Thinking-2507"
python -m hybrid_muon inspect "/path/to/Qwen3.5-4B"
python -m hybrid_muon inspect "/path/to/Qwen3.8-27B"
```

| Checkpoint | Muon tensors | AdamW decay tensors | AdamW no-decay tensors | Muon share of stored parameters |
| --- | ---: | ---: | ---: | ---: |
| Qwen3-4B-Thinking-2507 | 252 | 0 | 146 | 90.33% |
| Qwen3.5-4B | 207 | 173 | 358 | 78.81% |
| Qwen3.8-27B | 407 | 257 | 535 | 88.91% |

The Muon counts for Qwen3.5-4B and Qwen3.8-27B each include seven MTP projections stored in the checkpoint. Expand a model to see its full default output.

<details>
<summary>Qwen3-4B-Thinking-2507</summary>

```text
Checkpoint: Qwen3-4B-Thinking-2507 (stored tensors; assumes all trainable)
Group           Tensors       Parameters    Share   Default LR    Decay
muon                252    3,633,315,840  90.33%        5e-05    0.005
  model.layers.0.mlp.down_proj.weight  (2560, 9728)
  model.layers.0.mlp.gate_proj.weight  (9728, 2560)
adamw_decay           0                0   0.00%      1.5e-05     0.01
adamw_no_decay      146      389,152,256   9.67%      1.5e-05        0
  model.embed_tokens.weight  (151936, 2560)
  model.layers.0.input_layernorm.weight  (2560,)
```

</details>

<details>
<summary>Qwen3.5-4B</summary>

```text
Checkpoint: Qwen3.5-4B (stored tensors; assumes all trainable)
Group           Tensors       Parameters    Share   Default LR    Decay
muon                207    3,672,637,440  78.81%        5e-05    0.005
  model.language_model.layers.1.mlp.down_proj.weight  (2560, 9216)
  model.language_model.layers.1.mlp.gate_proj.weight  (9216, 2560)
adamw_decay         173      351,010,816   7.53%      1.5e-05     0.01
  model.language_model.layers.0.linear_attn.conv1d.weight  (8192, 1, 4)
  model.language_model.layers.0.linear_attn.in_proj_a.weight  (32, 2560)
adamw_no_decay      358      636,216,832  13.65%      1.5e-05        0
  model.language_model.embed_tokens.weight  (248320, 2560)
  model.language_model.layers.0.input_layernorm.weight  (2560,)
Review 24 possible fused projection(s), e.g. model.visual.blocks.0.attn.qkv.weight. Their names do not match the Muon rules; they use AdamW with decay.
MTP: 7 matching projections. For MTP training, inspect the model with its head attached.
```

</details>

<details>
<summary>Qwen3.8-27B</summary>

```text
Checkpoint: Qwen3.8-27B (stored tensors; assumes all trainable)
Group           Tensors       Parameters    Share   Default LR    Decay
muon                407   24,699,207,680  88.91%        5e-05    0.005
  mtp.layers.0.mlp.down_proj.weight  (5120, 17408)
  mtp.layers.0.mlp.gate_proj.weight  (17408, 5120)
adamw_decay         257      538,308,608   1.94%      1.5e-05     0.01
  mtp.fc.weight  (5120, 10240)
  model.language_model.layers.0.linear_attn.conv1d.weight  (10240, 1, 4)
adamw_no_decay      535    2,543,911,664   9.16%      1.5e-05        0
  lm_head.weight  (248320, 5120)
  mtp.layers.0.input_layernorm.weight  (5120,)
Review 27 possible fused projection(s), e.g. model.visual.blocks.0.attn.qkv.weight. Their names do not match the Muon rules; they use AdamW with decay.
MTP: 7 matching projections. For MTP training, inspect the model with its head attached.
```

</details>

## Training integrations

| Training setup | How to use the optimizer |
| --- | --- |
| TRL `SFTTrainer` on one GPU or with ordinary DDP | Pass `optimizers=(optimizer, None)`. Trainer creates the scheduler. |
| PyTorch DistributedDataParallel (DDP) | Build from `ddp_model.module` and run the forward pass through `ddp_model`. Each rank holds complete matrices and optimizer state. |
| DeepSpeed stage 0 | Pass the optimizer to the engine. Use BF16 autocast with FP32 parameters. |
| DeepSpeed ZeRO stages 1–3 | The standard wrapper passes flat parameter partitions to the optimizer. This package requires each complete 2D projection at `step()`. Stage 3 also shards model weights. [ZeRO stages](https://deepspeed.readthedocs.io/en/stable/zero3.html) |

[JamePeng](https://github.com/JamePeng) ran Qwen3 SFT with ZeRO-2 using a standalone training script with its own optimizer implementation. DeepSpeed also implements matrix-aware ZeRO updates for its own `MuonWithAuxAdam`. Enabling `zero_allow_untested_optimizer` only changes the optimizer check for a custom instance. [DeepSpeed ZeRO implementation](https://github.com/deepspeedai/DeepSpeed/blob/b5e0ba67f2757af1e2003fba9dc362db042a5ffe/deepspeed/runtime/zero/stage_1_and_2.py)

<details>
<summary><strong>TRL SFTTrainer</strong> · BF16 and per-group learning rates</summary>

Use the optimizer from the quick start with your model, tokenizer, and training dataset. Trainer handles gradient accumulation and optimizer steps. Its linear scheduler scales each group from its own initial learning rate. [TRL SFTTrainer](https://huggingface.co/docs/trl/sft_trainer)

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

For ordinary DDP, launch your script with one process per GPU, for example `torchrun --standalone --nproc-per-node=2 train_sft.py`. For TRL with DeepSpeed, use the stage 0 example below.

</details>

<details>
<summary><strong>PyTorch DDP</strong> · training step and checkpoints</summary>

Initialize the process group, place the model on the rank's GPU, and wrap it in DDP. Give each rank its own batch with a `DistributedSampler`. DDP synchronizes gradients during backward. On a BF16-capable GPU, keep the model parameters in FP32 and use autocast for the forward pass.

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

One rank can save `ddp_model.module.state_dict()` and `optimizer.state_dict()`. Restore both on every rank with the same trainable parameter names and group order. The two-process CPU/Gloo test compares three DDP steps against a single-process run on the combined batch. [DDP test](tests/test_optimizer.py)

</details>

<details>
<summary><strong>DeepSpeed stage 0</strong> · native engine and TRL configuration</summary>

Launch one process per BF16-capable GPU. Start with the unwrapped model on the rank's GPU in FP32 (in Transformers, pass `dtype=torch.float32` to `from_pretrained`). The engine and TRL examples share this configuration. DeepSpeed's `torch_autocast` enables BF16 computation while keeping the optimizer's original matrices. Omit the native `bf16`, `fp16`, and `optimizer` entries. [DeepSpeed mixed precision](https://deepspeed.readthedocs.io/en/stable/training.html#mixed-precision-training)

```python
ds_config = {
    "train_micro_batch_size_per_gpu": 1,
    "zero_optimization": {"stage": 0},
    "torch_autocast": {"enabled": True, "dtype": "bfloat16"},
}
```

**Native engine.** Pass the optimizer from the quick start and use the engine for backward and step.

```python
import deepspeed

engine, _, _, _ = deepspeed.initialize(model=model, optimizer=optimizer, config=ds_config)
loss = engine(**batch).loss
engine.backward(loss)
engine.step()
```

Call `engine.save_checkpoint(directory, tag=tag)` on every rank with the same tag. Resume with `engine.load_checkpoint(directory, tag=tag)`. [Checkpoint API](https://deepspeed.readthedocs.io/en/stable/model-checkpointing.html)

**TRL with DeepSpeed.** Override `create_optimizer` so Trainer can build the groups during engine initialization. Let Trainer fill in the batch settings and leave precision management to DeepSpeed. With the Accelerate launcher, select `--mixed_precision no`.

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

## Performance

**Compute.** In dense pretraining experiments spanning 399M to 1.5B non-embedding parameters, Liu et al. estimated that their Muon recipe reached the same loss using **52% of the training FLOPs** of tuned AdamW. This estimate comes from their compute-optimal scaling curves. Use `muon_config={"scale_mode": "paper"}` for the shape-dependent scaling studied in the paper. [Muon is Scalable for LLM Training](https://arxiv.org/html/2502.16982v1#S3.SS2)

**Memory.** At the same state precision, Muon saves one persistent buffer per selected matrix. It keeps one momentum buffer where AdamW keeps two moment buffers. Newton–Schulz iterations also use temporary memory and extra matrix multiplications. Qwen throughput and time to target loss remain unmeasured in this repository.

### Loop transformer experiments

During his loop transformer research, JamePeng compared an earlier hybrid implementation with AdamW on a **309M-parameter recurrent language model**. At the same matrix learning rate of `5e-5`, the hybrid run used **207.6 MiB less peak GPU memory**. Removing one FP32 buffer across the 54,525,952 selected parameters predicts a saving of **208.0 MiB**.

| Optimizer | Matrix LR | Training NLL (128 examples) | Validation NLL (32 new examples) | PyTorch peak allocated memory (GiB) |
| --- | ---: | ---: | ---: | ---: |
| AdamW | 5e-5 | 0.2312 | 6.0893 | 6.6365 |
| Hybrid (`paper`) | 2.5e-5 | 0.2575 | 5.6558 | 6.4337 |
| Hybrid (`paper`) | 3.5e-5 | 0.1638 | 5.7353 | 6.4337 |
| Hybrid (`paper`) | 5e-5 | 0.1131 | 5.8344 | 6.4337 |

All four runs used the same checkpoint, fresh optimizer state, and the same batch order on an RTX 5090 with FP32 parameters and BF16 autocast. Each ran for 500 steps on 1,162,894 supervised tokens. Parameters outside the 28 selected matrices used a learning rate of `1.5e-5`. NLL is token-weighted negative log-likelihood (lower is better). Experiment ID: `optimizer-lr-validation-2026-09-26`.

<details>
<summary>Model reports describing Muon and its variants</summary>

Pretraining examples include [Kimi K2](https://arxiv.org/html/2507.20534v1#S2.SS1), [Kimi K3](https://arxiv.org/html/2607.24653v1), [GLM-4.5 and GLM-5](https://arxiv.org/html/2602.15763v1), [Step 3.5 Flash](https://arxiv.org/html/2602.10604v1), [Inkling](https://thinkingmachines.ai/news/introducing-inkling/), [Qwen3.8-Flash-Next](https://arxiv.org/html/2608.30320v1), [DeepSeek-V4.1-Flash](https://arxiv.org/html/2609.19969v1), and [MAI Base 2](https://microsoft.ai/news/frontier-pre-training-at-microsoft-ai/). Xiaomi used the Muon variant Muown during mid-training. [MiMo-V2.6 report](https://arxiv.org/html/2610.11959v1)

</details>

## Author and license

JamePeng · <jame_peng@sina.com>

Hybrid Muon is licensed under [Apache-2.0](LICENSE). The Newton–Schulz iteration is adapted from Keller Jordan's Muon implementation. When redistributing copies or substantial portions of that code, include its copyright and permission notices. [Upstream source](https://github.com/KellerJordan/Muon/tree/f98f1cacc0263b04290753e32be8d498c1efc806) · [Full MIT notice](THIRD_PARTY_LICENSES.txt)

## How to cite

If you use Hybrid Muon in your work, please credit the project in your README, model card, or paper. Include the release version or commit you used so readers can identify the implementation.

> JamePeng. *Hybrid Muon*. [https://github.com/JamePeng/hybrid_muon](https://github.com/JamePeng/hybrid_muon).
