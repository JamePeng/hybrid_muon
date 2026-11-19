# Using Hybrid Muon with three dense Qwen models

Use `HybridMuonAdamW(model)` for all three checkpoints. The examples below show which tensors enter each group and how the counts change when you freeze the vision encoder or attach an MTP head.

## 1. Identify the model you are optimizing

| Checkpoint inspected | Transformer blocks in the text decoder | Other components |
| --- | --- | --- |
| Qwen3-4B-Thinking-2507 | 36 full-attention blocks, hidden size 2,560 | Text only. Input embeddings and output head share weights. |
| Qwen3.5-4B | 32 blocks, hidden size 2,560. Each set of four contains three linear-attention blocks and one full-attention block. | 24-block vision encoder, tied text embeddings, one checkpointed multi-token prediction (MTP) layer. |
| Qwen3.8-27B | 64 blocks, hidden size 5,120. The same pattern gives 48 linear-attention blocks and 16 full-attention blocks. | 27-block vision encoder, separate input embeddings and output head, one checkpointed MTP layer. |

**Qwen3.** Pass the full causal language model (`Qwen3ForCausalLM`). Each decoder block contributes four attention matrices (`q_proj`, `k_proj`, `v_proj`, `o_proj`) and three MLP matrices (`gate_proj`, `up_proj`, `down_proj`) to Muon. For example, `model.layers.0.self_attn.q_proj.weight` uses Muon and `model.layers.0.self_attn.q_norm.weight` uses AdamW without decay.

**Qwen3.5.** The full generation model exposes its text blocks under `model.language_model.layers`. In a linear-attention block, Muon takes `in_proj_qkv`, `in_proj_z`, and `out_proj` plus the three MLP matrices. AdamW with decay takes `in_proj_a.weight`, `in_proj_b.weight`, and `conv1d.weight`. Full-attention blocks contribute the same seven matrices as Qwen3. This gives **200 text-backbone Muon matrices**: 24 × 6 + 8 × 7.

**Qwen3.8-27B.** This checkpoint declares `model_type="qwen3_5"` and `Qwen3_5ForConditionalGeneration`. Its dense decoder uses the same projection paths as Qwen3.5-4B, giving **400 text-backbone Muon matrices**: 48 × 6 + 16 × 7. Both `lm_head.weight` and the separate token embedding use AdamW without decay. These assignments describe this package's defaults on the 27B checkpoint.

## 2. Choose the trainable modules before building the optimizer

Start with the model returned by your own loader, placed on its training device. For Qwen3, pass the causal language model directly. For Qwen3.5-4B and Qwen3.8-27B, pass the full conditional-generation model for vision-language training. Its trainable vision weights enter the AdamW groups. For text-only updates on that same wrapper, freeze its vision encoder first:

```python
model.model.visual.requires_grad_(False)  # Optional, for the two multimodal models
```

If you want **MTP training**, load the checkpoint's `mtp.*` tensors into a compatible head registered on the model before constructing the optimizer. Use the wrapper whose forward pass includes the MTP loss. Both multimodal checkpoints store 15 such tensors. Seven attention and MLP projections enter Muon, `mtp.fc.weight` enters AdamW with decay, and seven norm weights enter AdamW without decay. An MTP layer declared in the config only contributes parameters if the loaded model exposes it through `named_parameters()`.

## 3. Inspect the groups and take a step

```python
from hybrid_muon import HybridMuonAdamW

optimizer = HybridMuonAdamW(model, muon_config={"lr": 5e-5}, adamw_lr=1.5e-5)
for group in optimizer.param_groups:
    names = group["param_names"]
    mtp_count = sum("mtp" in name.split(".") for name in names)
    print(group["name"], len(names), "MTP:", mtp_count, "examples:", names[:2])
```

Use these counts to check your output. Each cell counts **parameter tensors**. The standard-wrapper columns use Transformers 5.8.0 model classes instantiated from the checkpoint configs on PyTorch's `meta` device, with weights tied according to config and all parameters trainable. The last three columns apply the optimizer to the names and shapes in the actual safetensors shard headers, including all stored MTP weights.

| Checkpoint | Wrapper Muon | Wrapper AdamW decay | Wrapper AdamW no decay | Checkpoint Muon | Checkpoint AdamW decay | Checkpoint AdamW no decay |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3-4B-Thinking-2507 | 252 | 0 | 146 | 252 | 0 | 146 |
| Qwen3.5-4B | 200 | 172 | 351 | 207 | 173 | 358 |
| Qwen3.8-27B | 400 | 256 | 528 | 407 | 257 | 535 |

For example, with the vision encoder frozen and MTP absent, Qwen3.8-27B yields **400 / 144 / 307** in the same group order. Qwen3.5-4B yields **200 / 72 / 154**. Empty AdamW groups are omitted. Check the printed names if your loader attaches different modules or freezes more parameters.

Given a batch on the model's device with inputs and labels for your training objective:

```python
optimizer.zero_grad(set_to_none=True)
loss = model(**batch).loss
loss.backward()
optimizer.step()
```

With ordinary DDP, finish attaching and freezing modules before wrapping. Build the optimizer from `ddp_model.module` and compute the loss through `ddp_model(**batch)`. Each rank needs memory for full parameters and its optimizer state.

The grouping checks used the full configurations and checkpoint headers. The training-step example was exercised on reduced-size, randomly initialized versions of all three configurations with text inputs on CPU.
