# Hybrid Muon

A PyTorch optimizer that applies Muon and AdamW to separate parameter groups. For worked examples, see [Using Hybrid Muon with Qwen3, Qwen3.5, and Qwen3.8-27B](docs/qwen-models.md).

[Package version](src/hybrid_muon/__init__.py) · Python 3.10+ · PyTorch 2.10+

## Install and use

From the repository root:

```sh
python -m pip install -e .
python -c "import hybrid_muon; print(hybrid_muon.__version__)"
```

Pass your model after attaching any optional modules and setting which parameters to train. The constructor accepts separate learning rates for Muon and AdamW.

```python
from hybrid_muon import HybridMuonAdamW

optimizer = HybridMuonAdamW(model, muon_config={"lr": 5e-5}, adamw_lr=1.5e-5)
for group in optimizer.param_groups:
    print(group["name"], group["lr"], group["param_names"])
```

Use `optimizer.zero_grad()` and `optimizer.step()` in your existing training loop.

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

## DDP and checkpoints

Use full parameter matrices on a single device or on each ordinary DistributedDataParallel (DDP) rank. Attach optional modules before wrapping the model in DDP, pass `ddp_model.module` to the optimizer, and run the forward pass through `ddp_model`.

Save `optimizer.state_dict()` alongside the model state. To resume, rebuild the same model and optimizer with the same trainable parameter names and group order, then call `optimizer.load_state_dict(saved_optimizer_state)`.

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
