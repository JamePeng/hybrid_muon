"""Read-only parameter grouping reports for live models and local checkpoints."""

import json
import math
from pathlib import Path

from .optimizer import DEFAULT_ADAMW_LR, DEFAULT_ADAMW_WEIGHT_DECAY, MUON_CONFIG, HybridMuonAdamW


def _report(title, tensors, verbose):
    settings = {
        "muon": (MUON_CONFIG["lr"], MUON_CONFIG["weight_decay"]),
        "adamw_decay": (DEFAULT_ADAMW_LR, DEFAULT_ADAMW_WEIGHT_DECAY),
        "adamw_no_decay": (DEFAULT_ADAMW_LR, 0.0),
    }
    groups = {group: [] for group in settings}
    frozen = frozen_muon = 0
    for name, shape, trainable in tensors:
        group = HybridMuonAdamW.parameter_group(name, len(shape))
        if trainable:
            groups[group].append((name, tuple(shape)))
        else:
            frozen += 1
            frozen_muon += group == "muon"

    sizes = {group: sum(math.prod(shape) for _, shape in rows) for group, rows in groups.items()}
    total = sum(sizes.values())
    lines = [title, "Group           Tensors       Parameters    Share   Default LR    Decay"]
    for group, (lr, decay) in settings.items():
        rows = groups[group]
        share = sizes[group] / total if total else 0
        lines.append(f"{group:15} {len(rows):7} {sizes[group]:16,} {share:7.2%} {lr:12g} {decay:8g}")
        for name, shape in (rows if verbose else rows[:2]):
            lines.append(f"  {name}  {shape}")
    if frozen:
        lines.append(f"Skipped {frozen} frozen tensors, including {frozen_muon} matching Muon projections.")
    if not groups["muon"]:
        lines.append("No trainable Muon projections. Check frozen weights, tensor ranks and projection names.")
    fused_names = {"qkv", "qkv_proj", "query_key_value", "c_attn"}
    fused = [
        name for name, shape in groups["adamw_decay"]
        if len(shape) == 2 and fused_names.intersection(name.lower().split("."))
    ]
    if fused:
        lines.append(f"Review {len(fused)} possible fused projection(s), e.g. {fused[0]}. "
                     "Their names do not match the Muon rules; they use AdamW with decay.")
    mtp = sum("mtp" in name.split(".") for name, _ in groups["muon"])
    if mtp:
        lines.append(f"MTP: {mtp} matching projections. For MTP training, inspect the model with its head attached.")
    return "\n".join(lines)


def inspect_model(model, *, verbose=False):
    """Return a grouping report for the model's current trainable parameters."""
    tensors = ((name, p.shape, p.requires_grad) for name, p in model.named_parameters())
    return _report(f"Model: {type(model).__name__} (shared parameters counted once)", tensors, verbose)


def _checkpoint_shapes(directory):
    from safetensors import SafetensorError, safe_open

    index = directory / "model.safetensors.index.json"
    shards = {}
    if index.is_file():
        weights = json.loads(index.read_text(encoding="utf-8"))["weight_map"]
        if not isinstance(weights, dict) or not weights:
            raise ValueError(f"Missing tensor entries in {index}")
        for name, filename in weights.items():
            shard = (directory / filename).resolve()
            if not shard.is_relative_to(directory.resolve()):
                raise ValueError(f"Shard outside checkpoint directory: {filename}")
            shards.setdefault(shard, []).append(name)
    else:
        shards[directory / "model.safetensors"] = None

    for path, names in shards.items():
        try:
            with safe_open(str(path), framework="pt", device="cpu") as shard:
                for name in sorted(shard.keys() if names is None else names):
                    yield name, shard.get_slice(name).get_shape(), True
        except SafetensorError as exc:
            raise ValueError(f"{path}: {exc}") from exc


def inspect_checkpoint(directory, *, verbose=False):
    """Preview the grouping of stored tensors without loading their values."""
    path = Path(directory)
    title = f"Checkpoint: {path.name} (stored tensors; assumes all trainable)"
    return _report(title, _checkpoint_shapes(path), verbose)
