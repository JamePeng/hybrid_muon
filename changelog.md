# Changelog

## 0.1.1 (2026-11-21)

- Preview parameter groups before training with `inspect_model(model)`, or inspect a local safetensors checkpoint with `python -m hybrid_muon inspect <directory>`. Both reports use the optimizer's grouping rules and show tensor counts, parameter shares, default learning rates, and weight decay. Use verbose mode to list every grouped tensor and its shape.
- Added inspection commands and sample output for Qwen3-4B-Thinking-2507, Qwen3.5-4B, and Qwen3.8-27B to the English and Simplified Chinese READMEs.

## 0.1.0 (2026-11-20)

- Introduced `HybridMuonAdamW` for dense Qwen-style models. It applies Muon to selected projection matrices and AdamW to the remaining trainable parameters, with separate learning rates and weight decay settings.
- Documented parameter grouping, checkpointing, and training with PyTorch DDP, TRL, and DeepSpeed.
