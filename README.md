# Hybrid Muon

A PyTorch optimizer that applies Muon and AdamW to separate parameter groups. For worked examples, see [Using Hybrid Muon with Qwen3, Qwen3.5, and Qwen3.8-27B](docs/qwen-models.md).

[Package version](src/hybrid_muon/__init__.py) · Python 3.10+ · PyTorch 2.10+

## Installation

### Option 1: Clone and install locally

```sh
git clone https://github.com/JamePeng/hybrid_muon.git
cd hybrid_muon
python -m pip install -e .
```

The editable install picks up changes you make in the clone.

### Option 2: Install directly from GitHub

```sh
python -m pip install "git+https://github.com/JamePeng/hybrid_muon.git"
```

After either installation, print the installed version:

```sh
python -c "import hybrid_muon; print(hybrid_muon.__version__)"
```

## Usage

Pass your model after attaching any optional modules and setting which parameters to train. The constructor accepts separate learning rates for Muon and AdamW.

```python
from hybrid_muon import HybridMuonAdamW

optimizer = HybridMuonAdamW(model, muon_config={"lr": 5e-5}, adamw_lr=1.5e-5)
for group in optimizer.param_groups:
    print(group["name"], group["lr"], group["param_names"])
```

Use `optimizer.zero_grad()` and `optimizer.step()` in your existing training loop.

## TRL SFTTrainer

For single-GPU or ordinary DDP training with an existing optimizer instance, pass `optimizers=(optimizer, None)` to `SFTTrainer` to let Trainer build the scheduler. For a script that also runs with DeepSpeed, override `create_optimizer` as shown below. This lets Trainer recreate the same parameter groups when initializing DeepSpeed. The example assumes an FP32 model, its tokenizer, a TRL training dataset, and a BF16-capable GPU.

```python
from trl import SFTConfig, SFTTrainer
from hybrid_muon import HybridMuonAdamW


class HybridSFTTrainer(SFTTrainer):
    def create_optimizer(self, model=None):
        if self.optimizer is None:
            self.optimizer = HybridMuonAdamW(
                self.model if model is None else model,
                muon_config={"lr": 5e-5},
                adamw_lr=1.5e-5,
            )
        return self.optimizer


args = SFTConfig(output_dir="checkpoints", bf16=True, lr_scheduler_type="linear")
trainer = HybridSFTTrainer(
    model=model,
    args=args,
    processing_class=tokenizer,
    train_dataset=train_dataset,
)
trainer.train()
```

Trainer handles gradient accumulation, scheduling, and optimizer steps. The linear scheduler scales both groups from their own initial learning rates. For ordinary DDP, launch the same script with one process per GPU, for example `torchrun --standalone --nproc-per-node=2 train_sft.py` for a user script named `train_sft.py`. At least one matching projection weight must be trainable. Adapter-only LoRA runs that freeze every matching weight will fail the Muon group check. [TRL SFTTrainer](https://huggingface.co/docs/trl/sft_trainer)

**With DeepSpeed**, use the same trainer subclass and replace the `args` assignment with the stage 0 configuration below. Set Trainer's precision flags explicitly so DeepSpeed's `torch_autocast` owns BF16 computation. When using Accelerate's launcher, select `--mixed_precision no` for this configuration.

```python
args = SFTConfig(
    output_dir="checkpoints",
    bf16=False,
    fp16=False,
    lr_scheduler_type="linear",
    deepspeed={
        "train_micro_batch_size_per_gpu": "auto",
        "gradient_accumulation_steps": "auto",
        "zero_optimization": {"stage": 0},
        "torch_autocast": {"enabled": True, "dtype": "bfloat16"},
    },
)
```

## Parameter groups

**Why split the parameters?** An attention or MLP projection mixes features through a matrix. Muon approximately orthogonalizes its momentum update, reducing how strongly a few singular directions dominate the step. A normalization vector scales individual features, and an embedding table stores a vector per token. This package uses AdamW's per-element adaptive updates for those parameters and lets you tune their learning rate separately from the projection matrices. [Original Muon design](https://kellerjordan.github.io/posts/muon/)

The optimizer reads `model.named_parameters()` once at construction, skips frozen parameters, and counts tied weights once.

| Group | Trainable parameters | Default weight decay |
| --- | --- | ---: |
| `muon` | 2D projection weights whose names contain `.layers.` and end in one of the suffixes below | 0.005 |
| `adamw_no_decay` | Remaining tensors of rank 0 or 1, or with `norm`, `bias`, `embed_tokens`, or `lm_head` in the name | 0 |
| `adamw_decay` | All other trainable parameters | 0.01 |

Muon matches the following projections by their `.weight` suffix:

- `self_attn`: `q_proj`, `k_proj`, `v_proj`, `o_proj`
- `mlp`: `gate_proj`, `up_proj`, `down_proj`
- `linear_attn` (Qwen3.5): `in_proj_qkv`, `in_proj_z`, `out_proj`

For **Qwen3.5 with multi-token prediction (MTP)**, pass the model that owns the attached MTP head. For example, `mtp.layers.0.self_attn.q_proj.weight` uses Muon, `mtp.fc.weight` uses AdamW with decay, and `mtp.norm.weight` uses AdamW without decay. For another architecture, inspect `param_names` to verify the assignments. Construction raises an error if no Muon weights match.

## Performance trade-offs

- **Compute to reach a target loss.** In dense language model pretraining experiments, Liu et al. fitted scaling curves for models with 399M to 1.5B non-embedding parameters. They estimated that their Muon recipe reached the same language-model loss using about **52% of the training FLOPs** of a tuned AdamW baseline. This estimate applies to the paper's compute-optimal pretraining setup. [Scaling study, section 3.2](https://arxiv.org/html/2502.16982v1#S3.SS2)
- **Optimizer memory and step time.** For each selected matrix, this implementation keeps one persistent momentum buffer. AdamW keeps two same-shaped moment buffers. This saves one matrix-sized buffer per Muon parameter at the same state precision. Newton-Schulz iterations also take temporary memory and extra matrix multiplications. Peak device memory, tokens per second, and time to reach a target loss on Qwen remain unbenchmarked in this repository.

### Optimizer experiments during loop transformer research

As part of his research on loop transformers, [JamePeng](https://github.com/JamePeng) used a 309M-parameter recurrent language model to explore how assigning selected weight matrices to Muon and the remaining parameters to AdamW affected training loss and GPU memory use.

The local experiment `optimizer-lr-validation-2026-09-26` tested an earlier implementation on an RTX 5090 with FP32 parameters and BF16 autocast. All four runs started from the same checkpoint with fresh optimizer state and saw the same 128 training examples in the same batch order. Each run completed 500 steps and processed 1,162,894 supervised tokens. The table reports token-weighted negative log-likelihood (NLL, lower is better) and peak GPU memory allocated by PyTorch. The learning rate for parameters outside the 28 selected matrices was fixed at `1.5e-5`.

| Optimizer | Matrix learning rate | Training NLL (128 examples) | Validation NLL (32 new examples) | Peak allocated memory (GiB) |
| --- | ---: | ---: | ---: | ---: |
| AdamW | 5e-5 | 0.2312 | 6.0893 | 6.6365 |
| Hybrid (`paper`) | 2.5e-5 | 0.2575 | 5.6558 | 6.4337 |
| Hybrid (`paper`) | 3.5e-5 | 0.1638 | 5.7353 | 6.4337 |
| Hybrid (`paper`) | 5e-5 | 0.1131 | 5.8344 | 6.4337 |

At the matched matrix learning rate of `5e-5`, Hybrid used **207.6 MiB less peak allocated memory**. The expected saving from one fewer FP32 buffer across 54,525,952 matrix parameters is **208.0 MiB**.

## Multi-GPU training

Muon orthogonalizes each projection's update as a complete 2D matrix. The choice of distributed strategy determines which tensors the optimizer sees at `step()`.

| Strategy | What this means for HybridMuonAdamW |
| --- | --- |
| PyTorch DistributedDataParallel (DDP) | Each rank holds the full model and optimizer state. DDP averages gradients before each rank applies the same update. |
| DeepSpeed ZeRO stage 0 | ZeRO is disabled. BF16 autocast runs eligible operations in BF16 while the optimizer updates complete FP32 matrices. |
| DeepSpeed ZeRO stages 1 and 2 | The standard wrapper passes flat parameter partitions to the underlying optimizer even though model weights are replicated. This package's `step()` requires the original 2D projection matrices. |
| DeepSpeed ZeRO stage 3 | Unsupported. Model parameters are partitioned as well. Gathering weights for the forward pass does not supply complete matrix updates at optimizer step time. |

**ZeRO-2 and Muon.** JamePeng has run Qwen3 SFT with ZeRO-2. The standalone training script defines its own optimizer; the packaged `HybridMuonAdamW` has a separate update implementation. DeepSpeed also has a ZeRO-aware path for its own `MuonWithAuxAdam`. That path handles matrix boundaries inside the ZeRO wrapper. Passing this package's optimizer to the standard wrapper does not select that path. [DeepSpeed ZeRO optimizer implementation](https://github.com/deepspeedai/DeepSpeed/blob/b5e0ba67f2757af1e2003fba9dc362db042a5ffe/deepspeed/runtime/zero/stage_1_and_2.py)

### PyTorch DDP

Launch one process per GPU and give each rank its own batch, for example with a `DistributedSampler`. After initializing the process group, move the model to the rank's GPU, attach or freeze any optional modules on that device, and wrap it in DDP. Construct `HybridMuonAdamW(ddp_model.module)` and compute the loss through `ddp_model(**batch)`. The repository's two-process CPU/Gloo test compares three DDP steps with a single-process run on the combined batch.

On BF16-capable GPUs, keep the model parameters in FP32 and enable BF16 autocast around the forward pass:

```python
optimizer.zero_grad(set_to_none=True)
with torch.autocast("cuda", dtype=torch.bfloat16):
    loss = ddp_model(**batch).loss
loss.backward()
optimizer.step()
```

For a replicated checkpoint, one rank can save `ddp_model.module.state_dict()` and `optimizer.state_dict()`. On resume, restore both on every rank using the same trainable parameter names and group order.

### DeepSpeed stage 0

Launch one process per BF16-capable GPU. Load the unwrapped model in FP32 (for Transformers, pass `dtype=torch.float32` to `from_pretrained`), move it to the rank's GPU, and pass the optimizer instance to `deepspeed.initialize`. The following configuration enables BF16 autocast for the forward pass and sets a per-GPU batch size of one.

```python
import deepspeed
from hybrid_muon import HybridMuonAdamW

optimizer = HybridMuonAdamW(model, muon_config={"lr": 5e-5}, adamw_lr=1.5e-5)
engine, _, _, _ = deepspeed.initialize(
    model=model,
    optimizer=optimizer,
    config={
        "train_micro_batch_size_per_gpu": 1,
        "zero_optimization": {"stage": 0},
        "torch_autocast": {"enabled": True, "dtype": "bfloat16"},
    },
)
loss = engine(**batch).loss
engine.backward(loss)
engine.step()
```

If adapting an existing DeepSpeed config, replace its native `bf16` or `fp16` block with the `torch_autocast` block above. Native precision modes can wrap the optimizer and flatten its matrices. Leave the JSON `optimizer` entry unset so the engine uses the supplied instance. [DeepSpeed mixed precision](https://deepspeed.readthedocs.io/en/stable/training.html#mixed-precision-training)

Setting `zero_allow_untested_optimizer` only changes DeepSpeed's optimizer check. It does not add support for sharded updates. For checkpointing, call `engine.save_checkpoint(directory, tag=tag)` on every rank with the same tag and use `engine.load_checkpoint(directory, tag=tag)` to resume. [DeepSpeed ZeRO stages](https://deepspeed.readthedocs.io/en/stable/zero3.html) · [Checkpoint API](https://deepspeed.readthedocs.io/en/stable/model-checkpointing.html)

## References

The implementation follows the original Muon update. Set `muon_config={"scale_mode": "paper"}` to use the shape-dependent scaling studied by Liu et al. [Original Muon write-up](https://kellerjordan.github.io/posts/muon/) · [Muon is Scalable for LLM Training](https://arxiv.org/html/2502.16982v1#S2.SS2)

<details>
<summary>Model reports describing Muon and its variants</summary>

Pretraining examples include [Kimi K2](https://arxiv.org/html/2507.20534v1#S2.SS1), [Kimi K3](https://arxiv.org/html/2607.24653v1), [GLM-4.5 and GLM-5](https://arxiv.org/html/2602.15763v1), [Step 3.5 Flash](https://arxiv.org/html/2602.10604v1), [Inkling](https://thinkingmachines.ai/news/introducing-inkling/), [Qwen3.8-Flash-Next](https://arxiv.org/html/2608.30320v1), [DeepSeek-V4.1-Flash](https://arxiv.org/html/2609.19969v1), and [MAI Base 2](https://microsoft.ai/news/frontier-pre-training-at-microsoft-ai/). Xiaomi used the Muon variant Muown during mid-training. [MiMo-V2.6 report](https://arxiv.org/html/2610.11959v1)

</details>

## Author

JamePeng (<jame_peng@sina.com>)

## License

[Apache-2.0](LICENSE). The Newton-Schulz iteration is adapted from [KellerJordan/Muon](https://github.com/KellerJordan/Muon/tree/f98f1cacc0263b04290753e32be8d498c1efc806) under the [MIT license](THIRD_PARTY_LICENSES.txt).
